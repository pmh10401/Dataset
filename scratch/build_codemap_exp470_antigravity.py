#!/usr/bin/env python3
"""Build and validate the versioned Exp470 Task 2 code-map triplet."""

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
PACKAGE_ANTIGRAVITY = "experiments/exp470_ecg200_spectral_entropy_numba_antigravity"
HTML_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp470_antigravity.html"
JSON_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp470_antigravity.json"
LOCK_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp470_antigravity.lock"
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp470_antigravity.html",
    "docs/codemap/codemap_exp470_antigravity.json",
    "docs/codemap/codemap_exp470_antigravity.lock",
)
CONFIG_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/config_antigravity.py"
SOURCES_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/sources_antigravity.py"
INTERPOLATION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/interpolation_antigravity.py"
SYNTHETIC_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/synthetic_antigravity.py"
PREDECESSOR_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/predecessor_antigravity.py"
CONTRACTS_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/contracts_antigravity.py"
CONFIG_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_config_antigravity.py"
CONTRACTS_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_contracts_antigravity.py"
PREDECESSOR_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_predecessor_antigravity.py"
SPECTRAL_ENTROPY_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/spectral_entropy_antigravity.py"
PYTHON_REFERENCE_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/python_reference_antigravity.py"
SPECTRAL_ENTROPY_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_spectral_entropy_antigravity.py"
PYTHON_REFERENCE_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_python_reference_antigravity.py"
ACCELERATION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/acceleration_antigravity.py"
ACCELERATION_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_acceleration_antigravity.py"
ASSESSMENT_ANTIGRAVITY = "scratch/exp470_acceleration_assessment_antigravity.json"
BUILDER_ANTIGRAVITY = "scratch/build_codemap_exp470_antigravity.py"
TASK_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/2026-08-24-exp470-ecg200-spectral-entropy-numba-antigravity.html/"
    "task-1-brief-antigravity.html"
)
TASK2_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/2026-08-24-exp470-ecg200-spectral-entropy-numba-antigravity.html/"
    "task-2-brief-antigravity.html"
)
TASK3_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/2026-08-24-exp470-ecg200-spectral-entropy-numba-antigravity.html/"
    "task-3-brief-antigravity.html"
)
IMPLEMENTATION_REPORT_ANTIGRAVITY = "docs/EXP470_IMPLEMENTATION_REPORT_antigravity.html"
TASK_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/2026-08-24-exp470-ecg200-spectral-entropy-numba-antigravity.html/"
    "task-1-report-antigravity.html"
)
TASK2_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/2026-08-24-exp470-ecg200-spectral-entropy-numba-antigravity.html/"
    "task-2-report-antigravity.html"
)
TASK3_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/2026-08-24-exp470-ecg200-spectral-entropy-numba-antigravity.html/"
    "task-3-report-antigravity.html"
)
TASK4_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/2026-08-24-exp470-ecg200-spectral-entropy-numba-antigravity.html/"
    "task-4-brief-antigravity.html"
)
TASK4_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/2026-08-24-exp470-ecg200-spectral-entropy-numba-antigravity.html/"
    "task-4-report-antigravity.html"
)
SCORING_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/scoring_antigravity.py"
QUALIFICATION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/qualification_antigravity.py"
EVALUATION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/evaluation_antigravity.py"
SCORING_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_scoring_antigravity.py"
QUALIFICATION_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_qualification_antigravity.py"
EVALUATION_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_evaluation_antigravity.py"
EXP468_PACKAGE_ANTIGRAVITY = "experiments/exp468_ecg200_local_stft_spectral_redistribution_antigravity"
EXP468_SOURCES_ANTIGRAVITY = f"{EXP468_PACKAGE_ANTIGRAVITY}/sources_antigravity.py"
EXP468_INTERPOLATION_ANTIGRAVITY = f"{EXP468_PACKAGE_ANTIGRAVITY}/interpolation_antigravity.py"
EXP468_SYNTHETIC_ANTIGRAVITY = f"{EXP468_PACKAGE_ANTIGRAVITY}/synthetic_antigravity.py"
EXP468_PREDECESSOR_ANTIGRAVITY = f"{EXP468_PACKAGE_ANTIGRAVITY}/predecessor_antigravity.py"
REPRESENTED_PATHS_ANTIGRAVITY = (
    TASK_BRIEF_ANTIGRAVITY,
    CONFIG_ANTIGRAVITY,
    SOURCES_ANTIGRAVITY,
    INTERPOLATION_ANTIGRAVITY,
    SYNTHETIC_ANTIGRAVITY,
    PREDECESSOR_ANTIGRAVITY,
    CONTRACTS_ANTIGRAVITY,
    CONFIG_TEST_ANTIGRAVITY,
    CONTRACTS_TEST_ANTIGRAVITY,
    PREDECESSOR_TEST_ANTIGRAVITY,
    SPECTRAL_ENTROPY_ANTIGRAVITY,
    PYTHON_REFERENCE_ANTIGRAVITY,
    SPECTRAL_ENTROPY_TEST_ANTIGRAVITY,
    PYTHON_REFERENCE_TEST_ANTIGRAVITY,
    ACCELERATION_ANTIGRAVITY,
    ACCELERATION_TEST_ANTIGRAVITY,
    ASSESSMENT_ANTIGRAVITY,
    BUILDER_ANTIGRAVITY,
    IMPLEMENTATION_REPORT_ANTIGRAVITY,
    TASK_REPORT_ANTIGRAVITY,
    TASK2_BRIEF_ANTIGRAVITY,
    TASK2_REPORT_ANTIGRAVITY,
    TASK3_BRIEF_ANTIGRAVITY,
    TASK3_REPORT_ANTIGRAVITY,
    TASK4_BRIEF_ANTIGRAVITY,
    TASK4_REPORT_ANTIGRAVITY,
    SCORING_ANTIGRAVITY,
    QUALIFICATION_ANTIGRAVITY,
    EVALUATION_ANTIGRAVITY,
    SCORING_TEST_ANTIGRAVITY,
    QUALIFICATION_TEST_ANTIGRAVITY,
    EVALUATION_TEST_ANTIGRAVITY,
    EXP468_SOURCES_ANTIGRAVITY,
    EXP468_INTERPOLATION_ANTIGRAVITY,
    EXP468_SYNTHETIC_ANTIGRAVITY,
    EXP468_PREDECESSOR_ANTIGRAVITY,
    *TRIPLET_ANTIGRAVITY,
)
HASHED_SOURCE_PATHS_ANTIGRAVITY = (
    CONFIG_ANTIGRAVITY,
    SOURCES_ANTIGRAVITY,
    INTERPOLATION_ANTIGRAVITY,
    SYNTHETIC_ANTIGRAVITY,
    PREDECESSOR_ANTIGRAVITY,
    CONTRACTS_ANTIGRAVITY,
    CONFIG_TEST_ANTIGRAVITY,
    CONTRACTS_TEST_ANTIGRAVITY,
    PREDECESSOR_TEST_ANTIGRAVITY,
    SPECTRAL_ENTROPY_ANTIGRAVITY,
    PYTHON_REFERENCE_ANTIGRAVITY,
    SPECTRAL_ENTROPY_TEST_ANTIGRAVITY,
    PYTHON_REFERENCE_TEST_ANTIGRAVITY,
    ACCELERATION_ANTIGRAVITY,
    ACCELERATION_TEST_ANTIGRAVITY,
    SCORING_ANTIGRAVITY,
    QUALIFICATION_ANTIGRAVITY,
    EVALUATION_ANTIGRAVITY,
    SCORING_TEST_ANTIGRAVITY,
    QUALIFICATION_TEST_ANTIGRAVITY,
    EVALUATION_TEST_ANTIGRAVITY,
    BUILDER_ANTIGRAVITY,
    EXP468_SOURCES_ANTIGRAVITY,
    EXP468_INTERPOLATION_ANTIGRAVITY,
    EXP468_SYNTHETIC_ANTIGRAVITY,
    EXP468_PREDECESSOR_ANTIGRAVITY,
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp470_task4_pair_repair_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp470_task4_pair_repair_codemap_lock_v1"
_VOID_TAGS_ANTIGRAVITY = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}


def _canonical_bytes_antigravity(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _fingerprint_antigravity(relative: str) -> dict[str, Any]:
    payload = (ROOT_ANTIGRAVITY / relative).read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


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


def _validate_html_document_antigravity(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    parser = _StrictHtmlAntigravity()
    parser.feed(text)
    parser.close()
    if parser.stack or not {"html", "head", "body"}.issubset(parser.seen):
        raise ValueError(f"invalid HTML document: {path}")
    return text


def _node_antigravity(node_id: str, path: str, caller: str, effect: str, tests: tuple[str, ...], symbol: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "path": path,
        "caller": caller,
        "effect": effect,
        "tests": list(tests),
        "evidence": {"path": path, "symbol": symbol},
    }


def _edge_antigravity(source: str, edge_type: str, target: str, path: str, symbol: str) -> dict[str, Any]:
    return {"from": source, "type": edge_type, "to": target, "evidence": {"path": path, "symbol": symbol}}


def _source_contains_antigravity(path: str, symbol: str) -> bool:
    return (ROOT_ANTIGRAVITY / path).is_file() and symbol in (ROOT_ANTIGRAVITY / path).read_text(encoding="utf-8")


def _imports_exact_authority_antigravity(
    source_path: str,
    authority_path: str,
    authority_symbol: str,
) -> bool:
    source = (ROOT_ANTIGRAVITY / source_path).read_text(encoding="utf-8")
    expected_module = authority_path.removesuffix(".py").replace("/", ".")
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == expected_module
        and any(alias.name == authority_symbol and alias.asname is None for alias in node.names)
        for node in ast.walk(ast.parse(source))
    )


def build_graph_antigravity() -> dict[str, Any]:
    nodes = [
        _node_antigravity("task1_brief", TASK_BRIEF_ANTIGRAVITY, "Task 1 implementer", "Bounds work to frozen contracts, exact Exp468 exports, and the versioned map; prohibits scoring, runs, and routing.", (), "Task 1"),
        _node_antigravity("config", CONFIG_ANTIGRAVITY, "contracts and later Exp470 stages", "Freezes seeds, geometry, E11/C11 method order, change order, epsilon, and required speedup.", (CONFIG_TEST_ANTIGRAVITY,), "Exp470ConfigAntigravity"),
        _node_antigravity("sources", SOURCES_ANTIGRAVITY, "later input stage", "Re-exports the exact Exp468 normal TRAIN loader without a wrapper.", (PREDECESSOR_TEST_ANTIGRAVITY,), "load_ecg200_normal_train_antigravity"),
        _node_antigravity("interpolation", INTERPOLATION_ANTIGRAVITY, "later input stage", "Re-exports the exact Exp468 96-to-140 interpolator without a wrapper.", (PREDECESSOR_TEST_ANTIGRAVITY,), "interpolate_to_140_antigravity"),
        _node_antigravity("synthetic", SYNTHETIC_ANTIGRAVITY, "later qualification stage", "Re-exports exact Exp468 primary synthetic rows without a wrapper.", (PREDECESSOR_TEST_ANTIGRAVITY,), "primary_rows_antigravity"),
        _node_antigravity("predecessor", PREDECESSOR_ANTIGRAVITY, "later C0 comparison", "Re-exports exact Exp468 C0 capture and scoring authorities without wrappers.", (PREDECESSOR_TEST_ANTIGRAVITY,), "score_c0_fold_antigravity"),
        _node_antigravity("contracts", CONTRACTS_ANTIGRAVITY, "later scorer and qualification", "Freezes typed fold evidence, ordered mappings, zero label-query governance, and later artifact digest shape.", (CONTRACTS_TEST_ANTIGRAVITY,), "Exp470QualificationAntigravity"),
        _node_antigravity("spectral_entropy", SPECTRAL_ENTROPY_ANTIGRAVITY, "later E11 qualification", "Computes direct-DFT five-window spectral entropy, LOO raw scores, ranks, and strict confirmation entirely in Numba kernels.", (SPECTRAL_ENTROPY_TEST_ANTIGRAVITY,), "score_spectral_entropy_numba_antigravity"),
        _node_antigravity("python_reference", PYTHON_REFERENCE_ANTIGRAVITY, "four-row scorer validation only", "Provides an independent math/statistics tuple oracle; production scoring and qualification do not import it.", (PYTHON_REFERENCE_TEST_ANTIGRAVITY, SPECTRAL_ENTROPY_TEST_ANTIGRAVITY), "score_spectral_entropy_python_reference_antigravity"),
        _node_antigravity("config_test", CONFIG_TEST_ANTIGRAVITY, "focused unittest command", "Covers exact frozen config values, immutability, and type/order drift.", (), "Exp470ConfigTestAntigravity"),
        _node_antigravity("contracts_test", CONTRACTS_TEST_ANTIGRAVITY, "focused unittest command", "Covers ordered immutable evidence, count/AUC bounds, seed-major folds, and governance boundaries.", (), "Exp470ContractsTestAntigravity"),
        _node_antigravity("predecessor_test", PREDECESSOR_TEST_ANTIGRAVITY, "focused unittest command", "Covers identity equality for every direct Exp468 authority export.", (), "Exp470PredecessorTestAntigravity"),
        _node_antigravity("spectral_entropy_test", SPECTRAL_ENTROPY_TEST_ANTIGRAVITY, "focused unittest command", "Covers Numba E11 direct-DFT oracle parity, normalization, strict boundaries, deterministic immutable outputs, validation, and compiled-only constraints.", (), "SpectralEntropyNumbaTestAntigravity"),
        _node_antigravity("python_reference_test", PYTHON_REFERENCE_TEST_ANTIGRAVITY, "focused unittest command", "Covers the independent four-row tuple oracle and its Python-math-only implementation boundary.", (), "PythonReferenceTestAntigravity"),
        _node_antigravity("acceleration", ACCELERATION_ANTIGRAVITY, "Task 3 runtime evidence and CLI manifest", "Measures Numba reference parity, byte determinism, cold/warm timings, and MLX capability only; selected backend remains Numba.", (ACCELERATION_TEST_ANTIGRAVITY,), "assess_runtime_antigravity"),
        _node_antigravity("acceleration_test", ACCELERATION_TEST_ANTIGRAVITY, "focused unittest command", "Covers fixed fixture digest, strict JSON, validation stops, Numba evidence, and MLX record presence.", (), "AccelerationAssessmentTestAntigravity"),
        _node_antigravity("assessment", ASSESSMENT_ANTIGRAVITY, "Task 3 runtime evidence CLI", "Records one finite JSON-only Numba/MLX assessment without qualification or dataset access.", (), "selected_backend"),
        _node_antigravity("implementation_report", IMPLEMENTATION_REPORT_ANTIGRAVITY, "Task 1 closeout", "Records only Task 1 implementation and verification state; does not report scoring or execution.", (), "Exp470 Task 1"),
        _node_antigravity("task_report", TASK_REPORT_ANTIGRAVITY, "Task 1 parent handoff", "Preserves RED/GREEN, map validation, self-review, and concern evidence.", (), "Exp470 Task 1"),
        _node_antigravity("task2_brief", TASK2_BRIEF_ANTIGRAVITY, "Task 2 implementer", "Bounds work to the single Numba E11 scorer and Python oracle; prohibits qualification, runs, MLX work, and routing.", (), "Task 2"),
        _node_antigravity("task2_report", TASK2_REPORT_ANTIGRAVITY, "Task 2 parent handoff", "Preserves RED/GREEN, code-map checks, self-review, and scorer-only concerns.", (), "Task 2"),
        _node_antigravity("task3_brief", TASK3_BRIEF_ANTIGRAVITY, "Task 3 implementer", "Bounds runtime evidence to the fixed smoke fixture; prohibits qualification, data access, and routing.", (), "Task 3"),
        _node_antigravity("task3_report", TASK3_REPORT_ANTIGRAVITY, "Task 3 parent handoff", "Preserves RED/GREEN runtime evidence, manifest, map validation, and scope boundary.", (), "Task 3"),
        _node_antigravity("task4_brief", TASK4_BRIEF_ANTIGRAVITY, "Task 4 implementer", "Bounds C0/E11/C11 scoring, pair-repair accounting, and synthetic-only qualification; prohibits data access and runs.", (), "Task 4"),
        _node_antigravity("scoring", SCORING_ANTIGRAVITY, "Task 4 qualification", "Calls exact C0 once, exact Numba E11 once, and combines C11 only in a compiled kernel.", (SCORING_TEST_ANTIGRAVITY,), "score_fold_methods_antigravity"),
        _node_antigravity("qualification", QUALIFICATION_ANTIGRAVITY, "Task 4 decision", "Uses compiled pair credit/counting and aggregates only ten synthetic source records.", (QUALIFICATION_TEST_ANTIGRAVITY,), "qualify_records_antigravity"),
        _node_antigravity("evaluation", EVALUATION_ANTIGRAVITY, "Task 4 decision result", "Applies only period repair, protected-break, normal-confirmation, fallback, and finite gates; AUC is observational.", (EVALUATION_TEST_ANTIGRAVITY,), "evaluate_repair_gates_antigravity"),
        _node_antigravity("scoring_test", SCORING_TEST_ANTIGRAVITY, "focused unittest command", "Covers compiled C11, exact fallback, frozen evidence, and Numba provenance.", (), "Exp470ScoringTestAntigravity"),
        _node_antigravity("qualification_test", QUALIFICATION_TEST_ANTIGRAVITY, "focused unittest command", "Covers compiled pair credit/counting, malformed vectors, blocked folds, and ten-record synthetic aggregation.", (), "Exp470QualificationTestAntigravity"),
        _node_antigravity("evaluation_test", EVALUATION_TEST_ANTIGRAVITY, "focused unittest command", "Covers pair-only decision gates and explicit AUC non-gating.", (), "Exp470EvaluationTestAntigravity"),
        _node_antigravity("task4_report", TASK4_REPORT_ANTIGRAVITY, "Task 4 parent handoff", "Preserves RED/GREEN, full verification, map validation, and scope evidence.", (), "Exp470 Task 4"),
        _node_antigravity("exp468_load_ecg200_normal_train", EXP468_SOURCES_ANTIGRAVITY, "Exp470 sources export", "Provides the exact reviewed Exp468 normal TRAIN loader object.", (), "load_ecg200_normal_train_antigravity"),
        _node_antigravity("exp468_interpolate_to_140", EXP468_INTERPOLATION_ANTIGRAVITY, "Exp470 interpolation export", "Provides the exact reviewed Exp468 96-to-140 interpolation object.", (), "interpolate_to_140_antigravity"),
        _node_antigravity("exp468_primary_rows", EXP468_SYNTHETIC_ANTIGRAVITY, "Exp470 synthetic export", "Provides the exact reviewed Exp468 primary synthetic-row object.", (), "primary_rows_antigravity"),
        _node_antigravity("exp468_capture_exp465_authority", EXP468_PREDECESSOR_ANTIGRAVITY, "Exp470 predecessor export", "Provides the exact reviewed Exp468 C0 authority-capture object.", (), "capture_exp465_authority_antigravity"),
        _node_antigravity("exp468_score_c0_fold", EXP468_PREDECESSOR_ANTIGRAVITY, "Exp470 predecessor export", "Provides the exact reviewed Exp468 C0 fold scorer object.", (), "score_c0_fold_antigravity"),
        _node_antigravity("map_builder", BUILDER_ANTIGRAVITY, "Task 3 generation and validation", "Deterministically writes or validates only the versioned Exp470 HTML/JSON/lock triplet, represented paths, source hashes, and graph evidence.", (), "build_graph_antigravity"),
    ]
    edges = [
        _edge_antigravity("contracts", "calls", "config", CONTRACTS_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("config_test", "tests", "config", CONFIG_TEST_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("contracts_test", "tests", "contracts", CONTRACTS_TEST_ANTIGRAVITY, "Exp470QualificationAntigravity"),
        _edge_antigravity("spectral_entropy", "calls", "config", SPECTRAL_ENTROPY_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("spectral_entropy_test", "tests", "spectral_entropy", SPECTRAL_ENTROPY_TEST_ANTIGRAVITY, "score_spectral_entropy_numba_antigravity"),
        _edge_antigravity("spectral_entropy_test", "tests", "python_reference", SPECTRAL_ENTROPY_TEST_ANTIGRAVITY, "score_spectral_entropy_python_reference_antigravity"),
        _edge_antigravity("python_reference_test", "tests", "python_reference", PYTHON_REFERENCE_TEST_ANTIGRAVITY, "score_spectral_entropy_python_reference_antigravity"),
        _edge_antigravity("acceleration", "calls", "spectral_entropy", ACCELERATION_ANTIGRAVITY, "score_spectral_entropy_numba_antigravity"),
        _edge_antigravity("acceleration", "calls", "python_reference", ACCELERATION_ANTIGRAVITY, "score_spectral_entropy_python_reference_antigravity"),
        _edge_antigravity("acceleration_test", "tests", "acceleration", ACCELERATION_TEST_ANTIGRAVITY, "assess_runtime_antigravity"),
        _edge_antigravity("predecessor_test", "tests", "sources", PREDECESSOR_TEST_ANTIGRAVITY, "load_ecg200_normal_train_antigravity"),
        _edge_antigravity("predecessor_test", "tests", "interpolation", PREDECESSOR_TEST_ANTIGRAVITY, "interpolate_to_140_antigravity"),
        _edge_antigravity("predecessor_test", "tests", "synthetic", PREDECESSOR_TEST_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("predecessor_test", "tests", "predecessor", PREDECESSOR_TEST_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("sources", "authority", "exp468_load_ecg200_normal_train", SOURCES_ANTIGRAVITY, "load_ecg200_normal_train_antigravity"),
        _edge_antigravity("interpolation", "authority", "exp468_interpolate_to_140", INTERPOLATION_ANTIGRAVITY, "interpolate_to_140_antigravity"),
        _edge_antigravity("synthetic", "authority", "exp468_primary_rows", SYNTHETIC_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("predecessor", "authority", "exp468_capture_exp465_authority", PREDECESSOR_ANTIGRAVITY, "capture_exp465_authority_antigravity"),
        _edge_antigravity("predecessor", "authority", "exp468_score_c0_fold", PREDECESSOR_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("scoring", "calls", "config", SCORING_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("scoring", "calls", "spectral_entropy", SCORING_ANTIGRAVITY, "score_spectral_entropy_numba_antigravity"),
        _edge_antigravity("scoring", "calls", "predecessor", SCORING_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("qualification", "calls", "scoring", QUALIFICATION_ANTIGRAVITY, "FoldMethodScoresAntigravity"),
        _edge_antigravity("qualification", "calls", "contracts", QUALIFICATION_ANTIGRAVITY, "Exp470QualificationAntigravity"),
        _edge_antigravity("evaluation", "calls", "contracts", EVALUATION_ANTIGRAVITY, "Exp470QualificationAntigravity"),
        _edge_antigravity("scoring_test", "tests", "scoring", SCORING_TEST_ANTIGRAVITY, "score_fold_methods_antigravity"),
        _edge_antigravity("qualification_test", "tests", "qualification", QUALIFICATION_TEST_ANTIGRAVITY, "qualify_records_antigravity"),
        _edge_antigravity("evaluation_test", "tests", "evaluation", EVALUATION_TEST_ANTIGRAVITY, "evaluate_repair_gates_antigravity"),
    ]
    return {
        "schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "represented_paths": list(REPRESENTED_PATHS_ANTIGRAVITY),
        "source_fingerprints": [_fingerprint_antigravity(path) for path in HASHED_SOURCE_PATHS_ANTIGRAVITY],
        "nodes": nodes,
        "edges": edges,
    }


def _render_html_antigravity(graph: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(node['id'])}</td><td>{html.escape(node['path'])}</td><td>{html.escape(node['caller'])}</td><td>{html.escape(node['effect'])}</td></tr>"
        for node in graph["nodes"]
    )
    embedded = _canonical_bytes_antigravity(graph).decode("utf-8").rstrip()
    return "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><title>Exp470 code map</title></head><body><main><h1>Exp470 Task 3 Code Map</h1><p>Versioned authority map; no generic alias is generated.</p><table><thead><tr><th>ID</th><th>Path</th><th>Caller</th><th>Effect</th></tr></thead><tbody>" + rows + "</tbody></table><script id=\"exp470-codemap-json-antigravity\" type=\"application/json\">" + embedded + "</script></main></body></html>\n"


def _lock_antigravity(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": LOCK_SCHEMA_ANTIGRAVITY,
        "graph_schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "graph_sha256": hashlib.sha256(_canonical_bytes_antigravity(graph)).hexdigest(),
        "represented_paths": graph["represented_paths"],
        "source_fingerprints": graph["source_fingerprints"],
        "triplet": list(TRIPLET_ANTIGRAVITY),
    }


def _validate_antigravity() -> None:
    graph = build_graph_antigravity()
    for path in REPRESENTED_PATHS_ANTIGRAVITY:
        if not (ROOT_ANTIGRAVITY / path).is_file():
            raise ValueError(f"represented path missing: {path}")
    for node in graph["nodes"]:
        evidence = node["evidence"]
        if not _source_contains_antigravity(evidence["path"], evidence["symbol"]):
            raise ValueError(f"node evidence does not resolve: {node['id']}")
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    if len(node_by_id) != len(graph["nodes"]):
        raise ValueError("code map contains duplicate node IDs")
    for edge in graph["edges"]:
        if edge["from"] not in node_by_id or edge["to"] not in node_by_id:
            raise ValueError(f"edge endpoint does not resolve: {edge['from']}->{edge['to']}")
        evidence = edge["evidence"]
        if not _source_contains_antigravity(evidence["path"], evidence["symbol"]):
            raise ValueError(f"edge evidence does not resolve: {edge['from']}->{edge['to']}")
        if edge["type"] == "authority":
            source_path = node_by_id[edge["from"]]["path"]
            target_evidence = node_by_id[edge["to"]]["evidence"]
            if evidence != {"path": source_path, "symbol": target_evidence["symbol"]}:
                raise ValueError(f"authority edge evidence is not an exact import: {edge['from']}->{edge['to']}")
            if not _imports_exact_authority_antigravity(
                source_path,
                target_evidence["path"],
                target_evidence["symbol"],
            ):
                raise ValueError(f"authority edge import does not resolve: {edge['from']}->{edge['to']}")
    expected_json = _canonical_bytes_antigravity(graph)
    if JSON_PATH_ANTIGRAVITY.read_bytes() != expected_json:
        raise ValueError("JSON code map is not deterministic or is stale")
    html_text = _validate_html_document_antigravity(HTML_PATH_ANTIGRAVITY)
    match = re.search(r'<script id="exp470-codemap-json-antigravity" type="application/json">(.*?)</script>', html_text, re.DOTALL)
    if match is None or _canonical_bytes_antigravity(json.loads(match.group(1))) != expected_json:
        raise ValueError("HTML embedded code map differs from JSON")
    if LOCK_PATH_ANTIGRAVITY.read_bytes() != _canonical_bytes_antigravity(_lock_antigravity(graph)):
        raise ValueError("code-map lock is not deterministic or is stale")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        _validate_antigravity()
        return
    graph = build_graph_antigravity()
    JSON_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(graph))
    HTML_PATH_ANTIGRAVITY.write_text(_render_html_antigravity(graph), encoding="utf-8")
    LOCK_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(_lock_antigravity(graph)))
    _validate_antigravity()


if __name__ == "__main__":
    main_antigravity()
