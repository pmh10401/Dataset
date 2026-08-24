#!/usr/bin/env python3
"""Build and validate the bounded Exp468 Task 5 publication code-map triplet."""

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
    "task-1-report-antigravity.html"
)
TASK2_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-2-report-antigravity.html"
)
TASK3_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-3-report-antigravity.html"
)
TASK3_FALLBACK_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-3-fallback-report-antigravity.html"
)
TASK4_BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-4-report-antigravity.html"
)
LOCAL_STFT_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/local_stft_antigravity.py"
ACCELERATION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/acceleration_antigravity.py"
BACKEND_SELECTION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/backend_selection_antigravity.py"
LOCAL_STFT_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_local_stft_antigravity.py"
ACCELERATION_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_acceleration_antigravity.py"
BACKEND_SELECTION_TEST_ANTIGRAVITY = (
    f"{PACKAGE_ANTIGRAVITY}/tests/test_backend_selection_antigravity.py"
)
BENCHMARK_ANTIGRAVITY = "scratch/benchmark_exp468_local_stft_acceleration_antigravity.py"
BENCHMARK_ARTIFACT_ANTIGRAVITY = "scratch/exp468_acceleration_benchmark_antigravity.json"
TASK3_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-3-report-antigravity.html"
)
TASK3_FALLBACK_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-3-fallback-report-antigravity.html"
)
TASK4_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-4-report-antigravity.html"
)
TASK5_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-5-report-antigravity.html"
)
SCORING_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/scoring_antigravity.py"
QUALIFICATION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/qualification_antigravity.py"
EVALUATION_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/evaluation_antigravity.py"
SCORING_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_scoring_antigravity.py"
QUALIFICATION_TEST_ANTIGRAVITY = (
    f"{PACKAGE_ANTIGRAVITY}/tests/test_qualification_antigravity.py"
)
EVALUATION_TEST_ANTIGRAVITY = (
    f"{PACKAGE_ANTIGRAVITY}/tests/test_evaluation_antigravity.py"
)
ARTIFACTS_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/artifacts_antigravity.py"
REPLAY_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/replay_antigravity.py"
DASHBOARD_EXPORT_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/dashboard_export_antigravity.py"
ARTIFACTS_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_artifacts_antigravity.py"
REPLAY_TEST_ANTIGRAVITY = f"{PACKAGE_ANTIGRAVITY}/tests/test_replay_antigravity.py"
DASHBOARD_EXPORT_TEST_ANTIGRAVITY = (
    f"{PACKAGE_ANTIGRAVITY}/tests/test_dashboard_export_antigravity.py"
)
RANK_AUC_ANTIGRAVITY = (
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/"
    "rank_auc_antigravity.py"
)
FALLBACK_AMENDMENT_ANTIGRAVITY = (
    "docs/superpowers/specs/"
    "2026-08-24-exp468-numpy-fallback-antigravity-design-amendment.html"
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
    LOCAL_STFT_TEST_ANTIGRAVITY,
    ACCELERATION_TEST_ANTIGRAVITY,
    BACKEND_SELECTION_TEST_ANTIGRAVITY,
    LOCAL_STFT_ANTIGRAVITY,
    ACCELERATION_ANTIGRAVITY,
    BACKEND_SELECTION_ANTIGRAVITY,
    BENCHMARK_ANTIGRAVITY,
    BENCHMARK_ARTIFACT_ANTIGRAVITY,
    TASK3_REPORT_ANTIGRAVITY,
    TASK3_FALLBACK_REPORT_ANTIGRAVITY,
    TASK4_REPORT_ANTIGRAVITY,
    TASK5_REPORT_ANTIGRAVITY,
    SCORING_ANTIGRAVITY,
    QUALIFICATION_ANTIGRAVITY,
    EVALUATION_ANTIGRAVITY,
    SCORING_TEST_ANTIGRAVITY,
    QUALIFICATION_TEST_ANTIGRAVITY,
    EVALUATION_TEST_ANTIGRAVITY,
    ARTIFACTS_ANTIGRAVITY,
    REPLAY_ANTIGRAVITY,
    DASHBOARD_EXPORT_ANTIGRAVITY,
    ARTIFACTS_TEST_ANTIGRAVITY,
    REPLAY_TEST_ANTIGRAVITY,
    DASHBOARD_EXPORT_TEST_ANTIGRAVITY,
    RANK_AUC_ANTIGRAVITY,
    FALLBACK_AMENDMENT_ANTIGRAVITY,
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
GRAPH_SCHEMA_ANTIGRAVITY = "exp468_task5_publication_codemap_v4"
LOCK_SCHEMA_ANTIGRAVITY = "exp468_task5_publication_codemap_lock_v4"
SOURCE_PARENT_COMMIT_ANTIGRAVITY = "d027cced7893a256a51978ce67dcd9590bfcbd35"
GENERATED_AT_ANTIGRAVITY = "2026-08-24T09:00:00+09:00"
PRE_APPROVAL_BENCHMARK_SHA256_ANTIGRAVITY = (
    "938d83b0319346270a6cc32b7b6ba3efb88122dc1bf431863d8030f4539fb63b"
)
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
            "Task 1",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task2_brief",
            TASK2_BRIEF_ANTIGRAVITY,
            "Caller: exact Task 2 implementation authority. Effect: fixes adapter/test ownership, exact Exp466 imports, TRAIN/TEST boundary, four-scope C0 byte parity, RED/GREEN sequence, dashboard/map updates, verification commands, and report contract.",
            "Task 2",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_brief",
            TASK3_BRIEF_ANTIGRAVITY,
            "Caller: exact Task 3 implementation authority. Effect: fixes the local-STFT/JS formula, LOO and full-fit references, rank and confirmation boundaries, immutable score contract, backend parity, complete 10-fold workload, cold acceleration gate, blocked-state truth, owned files, verification, and report contract.",
            "Task 3",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_fallback_brief",
            TASK3_FALLBACK_BRIEF_ANTIGRAVITY,
            "Caller: exact Task 3 fallback closure authority. Effect: requires selection-contract RED/GREEN, immutable gate/fallback separation, approval-document and pre-approval benchmark hashes, checked Task 3 truth, zero/null execution boundaries, code-map updates, verification, report, and commit.",
            "fallback",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task4_brief",
            TASK4_BRIEF_ANTIGRAVITY,
            "Caller: exact Task 4 implementation authority. Effect: fixes test-first ownership, approved NumPy decision consumption, no backend selector, exact blocked folds, immutable C0/S10/C10 evidence, raw-input independent replay, fold-local AUC, every frozen per-fold and ten-fold gate, scalar-only evaluation, zero private/official execution, verification, report, and commit requirements.",
            "Task 4",
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
            "Caller: Task-by-Task controller and later implementers. Effect: records Task 1-5 completion, the unchanged failed 1.1x gate, approved NumPy fallback, exact scoring/replay/publication/gate definitions, zero/null execution truth, and Task 6 as the next implementation boundary.",
            "implementation_task_5_complete_no_private_run",
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
            "Caller: Task 4 qualification/evaluation and Task 5 publication/replay. Effect: supplies immutable fold and ten-row qualification containers while enforcing exact float AUC shape/range, seed-major order, immutable count mappings, reason agreement, zero label queries, and null pre-run artifact hash without calculating the scientific gate.",
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
            "task3_local_stft_test",
            LOCAL_STFT_TEST_ANTIGRAVITY,
            "Caller: Task 3 RED contract and later regression suite. Effect: specifies exact 5x28 framing, independent DFT distribution oracle, DC/Nyquist behavior, invariances, natural-log JS, LOO/full-fit references, leakage rejection, <= rank ties, strict confirmation, validation, and immutability before production modules exist.",
            "LocalStftTransformAntigravityTest",
            [LOCAL_STFT_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_acceleration_test",
            ACCELERATION_TEST_ANTIGRAVITY,
            "Caller: Task 3 RED contract and later regression suite. Effect: specifies NumPy-oracle versus sequential Numba full-output parity, finite/raw/rank/confirmation equality, fixed 1.1x gate truth, workload counts, backend assessment structure, and invalid-shape rejection before the acceleration module exists.",
            "AccelerationParityAntigravityTest",
            [ACCELERATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_backend_selection_test",
            BACKEND_SELECTION_TEST_ANTIGRAVITY,
            "Caller: Task 3 fallback RED/GREEN suite and audit preflight. Effect: detects approval bypass, gate-condition omission, missing non-NumPy failure reasons, malformed approval SHA-256, mutable decisions, and drift between the production decision and approval-linked benchmark artifact.",
            "BackendSelectionAntigravityTest",
            [BACKEND_SELECTION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_local_stft",
            LOCAL_STFT_ANTIGRAVITY,
            "Caller: Task 3 focused tests, acceleration backends, and future Task 4 scoring. Effect: implements the frozen 5x28 float64 STFT distribution, natural-log JS, fit LOO/query full-fit references, <= rank, strict confirmation, and immutable validated spectral-score result without C10/AUC/qualification logic.",
            "score_local_stft_antigravity",
            [LOCAL_STFT_TEST_ANTIGRAVITY, ACCELERATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_acceleration",
            ACCELERATION_ANTIGRAVITY,
            "Caller: Task 3 acceleration tests and full-workload benchmark. Effect: runs shared-calibration combined query batches, sequential fastmath-disabled Numba direct DFT, MLX float64 capability rejection, exact parity/determinism checks, and the unweakened cold 1.1x selection gate.",
            "assess_acceleration_antigravity",
            [ACCELERATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_backend_selection",
            BACKEND_SELECTION_ANTIGRAVITY,
            "Caller: the fallback behavior suite now and future Task 4 scorer. Effect: binds an unchanged AccelerationAssessmentAntigravity to an immutable operational decision, requiring the full non-NumPy gate for acceleration or audited approval SHA and candidate failure reasons for NumPy fallback.",
            "finalize_backend_decision_antigravity",
            [BACKEND_SELECTION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_benchmark",
            BENCHMARK_ANTIGRAVITY,
            "Caller: Task 3 operator. Effect: assembles but does not time the TRAIN-only seed-major ten-fold normal plus four-change workload from exact Task 2 adapters and Exp466 folds, then times the complete S10 workload sequentially and emits source-backed blocked evidence.",
            "build_workload_antigravity",
            [ACCELERATION_TEST_ANTIGRAVITY, BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_benchmark_artifact",
            BENCHMARK_ARTIFACT_ANTIGRAVITY,
            "Caller: backend selection test, Task 3 reports, implementation report, dashboard, code map, and controller review. Effect: preserves versions/device, exact 552/690/1242/6210 counts, every timing and parity value, gate_passed=false, while adding fallback_authorized=true, selected_backend=numpy, approval document SHA, pre-approval bytes SHA, zero runs, and null results/hashes.",
            "TASK_3_COMPLETE_NUMPY_FALLBACK_AUTHORIZED",
            [BACKEND_SELECTION_TEST_ANTIGRAVITY, BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_report",
            TASK3_REPORT_ANTIGRAVITY,
            "Caller: controller and final reviewer. Effect: records RED/GREEN, exact formula/LOO/leakage evidence, complete benchmark table, parity/fallback/profile truth, unchecked Task 3/Step 5, zero/null execution truth, self-review, commit identity convention, and concern.",
            "BLOCKED_AT_ACCELERATION_GATE",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_fallback_amendment",
            FALLBACK_AMENDMENT_ANTIGRAVITY,
            "Caller: user approval and fallback decision. Effect: authorizes NumPy for Task 4 without changing the failed acceleration gate, benchmark values, scientific formula, inputs, seeds/folds, workload, tolerance, confirmation rule, or zero/null execution truth.",
            "승인 결론",
            [BACKEND_SELECTION_TEST_ANTIGRAVITY, BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task3_fallback_report",
            TASK3_FALLBACK_REPORT_ANTIGRAVITY,
            "Caller: controller and final reviewer. Effect: records fallback RED/GREEN, owned changes, preserved benchmark values, checked Task 3 truth, exact verification commands/counts, enclosing commit convention, and concrete concerns.",
            "Exp468 Task 3 NumPy fallback closure 보고서",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task4_scoring_test",
            SCORING_TEST_ANTIGRAVITY,
            "Caller: Task 4 RED/GREEN suite. Effect: detects approval-decision drift, backend-selector exposure, combined-query/calibration drift, wrong C0/S10/C10 assembly, non-strict confirmation or C0 fallback, and mutable or aliased evidence.",
            "ScoringAntigravityTest",
            [SCORING_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task4_qualification_test",
            QUALIFICATION_TEST_ANTIGRAVITY,
            "Caller: Task 4 RED/GREEN suite. Effect: detects wrong fold arrays/order/coverage, TRAIN-byte drift, stored-score tampering across C0/raw/rank/confirmation/C10, pooled or mismatched AUC, missing exact boundaries/reasons, mean gates, and mutable results.",
            "QualificationAntigravityTest",
            [QUALIFICATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task4_evaluation_test",
            EVALUATION_TEST_ANTIGRAVITY,
            "Caller: Task 4 RED/GREEN suite. Effect: detects non-scalar recomputation, incorrect 10-of-10 or seed-normal gate aggregation, premature final PASS wording, routing/promotion mutation, TEST-label use, and mutable decisions.",
            "EvaluationAntigravityTest",
            [EVALUATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task4_scoring",
            SCORING_ANTIGRAVITY,
            "Caller: Task 4 scoring tests and fold-fixture assembler. Effect: rehydrates the checked Task 3 assessment, verifies approval SHA 6780edb..., binds the immutable NumPy decision without a public backend selector, scores C0 plus one shared-calibration S10 batch, constructs strict-confirmation C10, and retains owned read-only raw replay inputs/evidence.",
            "score_fold_methods_antigravity",
            [SCORING_TEST_ANTIGRAVITY, QUALIFICATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task4_qualification",
            QUALIFICATION_ANTIGRAVITY,
            "Caller: Task 4 qualification tests and Task 5 publication runner. Effect: enforces exact 14/14/14/14/13 blocked folds for both seeds, exact shared 69x140 TRAIN bytes and OOF coverage, independently replays raw C0/S10/C10, computes same-fold Mann-Whitney AUC, evaluates every frozen per-fold gate with deterministic reasons, and emits exactly ten immutable scalar rows without a mean gate.",
            "qualify_antigravity",
            [QUALIFICATION_TEST_ANTIGRAVITY, EVALUATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task4_evaluation",
            EVALUATION_ANTIGRAVITY,
            "Caller: Task 4 evaluation tests and Task 5 publication. Effect: consumes scalar qualification rows only, requires C10 10-of-10 and zero normal confirmations per seed, returns only awaiting-private or blocked-before-private status, and freezes all routing/champion/pending/promotion/TEST-label effects false.",
            "evaluate_qualification_antigravity",
            [EVALUATION_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task4_report",
            TASK4_REPORT_ANTIGRAVITY,
            "Caller: controller and final reviewer. Effect: records module-absence RED, focused/full/lower GREEN counts, approved NumPy SHA, exact replay and gate behavior, bounded TRAIN-only fixture outcome, portable dashboard receipt, zero/null execution truth, commit convention, and concerns.",
            "C0/S10/C10와 10-fold 자격 판정 구현 보고서",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task5_artifacts",
            ARTIFACTS_ANTIGRAVITY,
            "Caller: Task 5 publication tests and Task 6 runner. Effect: replays Task 4 qualification/decision, preserves exact raw evidence and provenance, and writes only the canonical seven-leaf result root with self-excluding SHA-256 digests.",
            "publish_result_antigravity",
            [ARTIFACTS_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task5_replay",
            REPLAY_ANTIGRAVITY,
            "Caller: Task 5 replay tests and Task 6 inspect path. Effect: reads only seven leaves, rejects canonical/hash/schema drift, reconstructs source-free records, and independently recomputes Task 4 C0/S10/C10, AUC, qualification, and decision.",
            "inspect_result_antigravity",
            [REPLAY_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task5_dashboard_export",
            DASHBOARD_EXPORT_ANTIGRAVITY,
            "Caller: Task 5 dashboard test and portable artifact workflow. Effect: renders deterministic self-contained Korean HTML with per-fold period evidence, protected-change deltas, normal confirmation errors, qualification state, and both private runs not run.",
            "export_dashboard_antigravity",
            [DASHBOARD_EXPORT_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task5_artifacts_test",
            ARTIFACTS_TEST_ANTIGRAVITY,
            "Caller: Task 5 RED/GREEN publication suite. Effect: detects seven-leaf/self-excluding hash drift, canonical JSON/newline drift, provenance and record-order/count drift, contract-boundary tampering before root creation, and non-deterministic duplicate roots.",
            "ArtifactsAntigravityTest",
            [ARTIFACTS_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task5_replay_test",
            REPLAY_TEST_ANTIGRAVITY,
            "Caller: Task 5 RED/GREEN replay suite. Effect: detects source-loader access, raw/fit/rank/confirmation/C10/AUC/delta/normal/decision/context/summary/failed-fold/dashboard/provenance tampering, stale or repaired hashes, malformed leaves, duplicate keys, and extra leaves.",
            "ReplayAntigravityTest",
            [REPLAY_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task5_dashboard_export_test",
            DASHBOARD_EXPORT_TEST_ANTIGRAVITY,
            "Caller: Task 5 RED/GREEN dashboard suite. Effect: detects non-deterministic output and omission of period, improvement, normal-error, qualification PASS/FAIL private status, both private not_run states, and checked Task 5 evidence.",
            "DashboardExportAntigravityTest",
            [DASHBOARD_EXPORT_TEST_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task5_report",
            TASK5_REPORT_ANTIGRAVITY,
            "Caller: controller and final reviewer. Effect: records Task 5 RED/GREEN evidence, seven-leaf contract, independent replay tamper evidence, portable HTML receipt, zero private/official execution truth, and the Task 6 checkpoint.",
            "Task 5 구현 보고서",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task6_cli",
            PLAN_ANTIGRAVITY,
            "Caller: future Task 6 CLI evaluate/inspect boundary. Effect: consumes the Task 5 publication and replay contracts without changing frozen science or exposing configuration selectors.",
            "cli_antigravity.py",
            [PLAN_ANTIGRAVITY],
        ),
        _node_antigravity(
            "task6_pipeline",
            PLAN_ANTIGRAVITY,
            "Caller: future Task 6 pipeline runner. Effect: consumes the Task 5 seven-leaf publication and independent inspect result as the only result boundary before a user-approved private run.",
            "pipeline",
            [PLAN_ANTIGRAVITY],
        ),
        _node_antigravity(
            "rank_auc",
            RANK_AUC_ANTIGRAVITY,
            "Caller: Task 4 qualification through the reviewed exact import. Effect: supplies the unchanged Mann-Whitney pair-counting AUC authority for each normal-versus-change pair within one fold.",
            "mann_whitney_auc_antigravity",
            [QUALIFICATION_TEST_ANTIGRAVITY],
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
            "Caller: governance and scientific reviewers. Effect: preserves the historical Task 1 freeze while recording Task 5's frozen seven-leaf publication/replay contract, backend-neutral science, and zero score/private/official/label activity plus null results and hashes.",
            "implementation_task_5_complete_no_private_run",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "runbook",
            RUNBOOK_ANTIGRAVITY,
            "Caller: Task 5 final verifier and Task 6 operator. Effect: provides focused/full/lower, code-map, portable-HTML, JSON, and diff verification for the completed publication boundary, states that private/official execution is not run, and limits the next step to the Task 6 checkpoint.",
            "Task 5",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "implementation_report",
            IMPLEMENTATION_REPORT_ANTIGRAVITY,
            "Caller: Task 1-5 closeout and later audit. Effect: records the frozen scientific/parity evidence, immutable NumPy decision, exact fold/replay/AUC/gate/publication implementation, focused/full/lower verification, Task 5 completion, Task 6 checkpoint, and zero/null official pre-run truth.",
            "Task 5",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "dashboard_artifact",
            DASHBOARD_ARTIFACT_ANTIGRAVITY,
            "Caller: portable dashboard generator. Effect: stores source-backed Task 5 publication/replay/dashboard completion truth, unchanged failed acceleration gate, approved NumPy fallback, exact verification counts, PASS/FAIL qualification and private-awaiting/blocked status, zero official/private runs, null results, and checklist state.",
            "implementation_summary",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "dashboard",
            DASHBOARD_ANTIGRAVITY,
            "Caller: user-facing progress review. Effect: renders Task 5 scoring/replay/publication/gate completion, the approved NumPy fallback and still-failed acceleration gate, exact verification counts, checked Task 5, the Task 6 checkpoint, and zero/null/no-TEST truth.",
            "Task 5",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "map_builder",
            BUILDER_ANTIGRAVITY,
            "Caller: every later Exp468 code-changing task preflight and final reviewer. Effect: generates the canonical HTML/JSON/lock triplet or validates Task 1-5 structure, exact scoring/qualification/evaluation/publication/replay/dashboard dependencies, immutable approval evidence, exact benchmark preservation, embedded JSON, dashboard truth, and all tracked fingerprints without writing in validate-only mode.",
            "build_triplet_antigravity",
            [BUILDER_ANTIGRAVITY],
        ),
    ]
    edges = [
        _edge_antigravity("task1_brief", "governs", "task1_config", TASK1_BRIEF_ANTIGRAVITY, "Task 1"),
        _edge_antigravity("task1_brief", "governs", "task1_contracts", TASK1_BRIEF_ANTIGRAVITY, "Task 1"),
        _edge_antigravity("approved_design", "governs", "task1_config", DESIGN_ANTIGRAVITY, "고정 입력과 범위"),
        _edge_antigravity("implementation_plan", "governs", "task1_config", PLAN_ANTIGRAVITY, "고정 설정·결과 계약·문서 골격"),
        _edge_antigravity("implementation_plan", "governs", "task1_contracts", PLAN_ANTIGRAVITY, "Exp468QualificationAntigravity"),
        _edge_antigravity("task1_config", "governs", "task1_contracts", CONFIG_ANTIGRAVITY, "change_types"),
        _edge_antigravity("task1_config_test", "tests", "task1_config", CONFIG_TEST_ANTIGRAVITY, "test_frozen_stft_authority_antigravity"),
        _edge_antigravity("task1_contracts_test", "tests", "task1_contracts", CONTRACTS_TEST_ANTIGRAVITY, "test_fold_auc_tuples_are_exact_finite_and_positionally_bound_antigravity"),
        _edge_antigravity("task2_brief", "governs", "task2_sources", TASK2_BRIEF_ANTIGRAVITY, "Task 2"),
        _edge_antigravity("task2_brief", "governs", "task2_interpolation", TASK2_BRIEF_ANTIGRAVITY, "Task 2"),
        _edge_antigravity("task2_brief", "governs", "task2_synthetic", TASK2_BRIEF_ANTIGRAVITY, "Task 2"),
        _edge_antigravity("task2_brief", "governs", "task2_predecessor", TASK2_BRIEF_ANTIGRAVITY, "Task 2"),
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
        _edge_antigravity("task3_brief", "governs", "task3_local_stft_test", TASK3_BRIEF_ANTIGRAVITY, "Task 3"),
        _edge_antigravity("task3_brief", "governs", "task3_acceleration_test", TASK3_BRIEF_ANTIGRAVITY, "Task 3"),
        _edge_antigravity("task3_fallback_brief", "governs", "task3_backend_selection_test", TASK3_FALLBACK_BRIEF_ANTIGRAVITY, "fallback"),
        _edge_antigravity("task3_fallback_brief", "governs", "task3_backend_selection", TASK3_FALLBACK_BRIEF_ANTIGRAVITY, "fallback"),
        _edge_antigravity("task3_fallback_brief", "governs", "task3_fallback_amendment", TASK3_FALLBACK_BRIEF_ANTIGRAVITY, "fallback"),
        _edge_antigravity("task3_fallback_brief", "governs", "task3_benchmark_artifact", TASK3_FALLBACK_BRIEF_ANTIGRAVITY, "fallback"),
        _edge_antigravity("implementation_plan", "governs", "task3_local_stft_test", PLAN_ANTIGRAVITY, "Task 3 — S10 국소 STFT·JS 점수와 Numba/MLX 가속"),
        _edge_antigravity("implementation_plan", "governs", "task3_acceleration_test", PLAN_ANTIGRAVITY, "Step 5 — NumPy/Numba/MLX 동일성·속도 gate"),
        _edge_antigravity("task3_brief", "governs", "task3_local_stft", TASK3_BRIEF_ANTIGRAVITY, "고정 S10 수식"),
        _edge_antigravity("task3_brief", "governs", "task3_acceleration", TASK3_BRIEF_ANTIGRAVITY, "실제 10-fold workload와 backend"),
        _edge_antigravity("implementation_plan", "governs", "task3_local_stft", PLAN_ANTIGRAVITY, "Task 3 — S10 국소 STFT·JS 점수와 Numba/MLX 가속"),
        _edge_antigravity("implementation_plan", "governs", "task3_acceleration", PLAN_ANTIGRAVITY, "Step 5 — NumPy/Numba/MLX 동일성·속도 gate"),
        _edge_antigravity("task1_config", "governs", "task3_local_stft", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task3_local_stft_test", "tests", "task3_local_stft", LOCAL_STFT_TEST_ANTIGRAVITY, "LocalStftScoringAntigravityTest"),
        _edge_antigravity("task3_acceleration_test", "tests", "task3_acceleration", ACCELERATION_TEST_ANTIGRAVITY, "AccelerationParityAntigravityTest"),
        _edge_antigravity("task3_backend_selection_test", "tests", "task3_backend_selection", BACKEND_SELECTION_TEST_ANTIGRAVITY, "finalize_backend_decision_antigravity"),
        _edge_antigravity("task3_backend_selection_test", "tests", "task3_benchmark_artifact", BACKEND_SELECTION_TEST_ANTIGRAVITY, "pre_approval_benchmark_sha256"),
        _edge_antigravity("task3_acceleration", "calls", "task3_local_stft", ACCELERATION_ANTIGRAVITY, "score_local_stft_antigravity"),
        _edge_antigravity("task3_backend_selection", "binds", "task3_acceleration", BACKEND_SELECTION_ANTIGRAVITY, "AccelerationAssessmentAntigravity"),
        _edge_antigravity("task3_fallback_amendment", "authorizes", "task3_backend_selection", FALLBACK_AMENDMENT_ANTIGRAVITY, "gate_passed=false"),
        _edge_antigravity("task3_benchmark", "calls", "task3_acceleration", BENCHMARK_ANTIGRAVITY, "assess_acceleration_antigravity"),
        _edge_antigravity("task3_benchmark", "generates", "task3_benchmark_artifact", BENCHMARK_ANTIGRAVITY, "assessment_payload_antigravity"),
        _edge_antigravity("task3_benchmark_artifact", "informs", "task3_report", BENCHMARK_ARTIFACT_ANTIGRAVITY, '"gate_passed": false'),
        _edge_antigravity("task3_report", "informs", "implementation_report", TASK3_REPORT_ANTIGRAVITY, "Self-review·commit·concerns"),
        _edge_antigravity("task3_benchmark_artifact", "informs", "task3_fallback_report", BENCHMARK_ARTIFACT_ANTIGRAVITY, "TASK_3_COMPLETE_NUMPY_FALLBACK_AUTHORIZED"),
        _edge_antigravity("task3_fallback_report", "informs", "implementation_report", TASK3_FALLBACK_REPORT_ANTIGRAVITY, "변경 파일"),
        _edge_antigravity("task3_benchmark_artifact", "informs", "dashboard_artifact", BENCHMARK_ARTIFACT_ANTIGRAVITY, "selected_backend"),
        _edge_antigravity("task4_brief", "governs", "task4_scoring_test", TASK4_BRIEF_ANTIGRAVITY, "Task 4"),
        _edge_antigravity("task4_brief", "governs", "task4_qualification_test", TASK4_BRIEF_ANTIGRAVITY, "Task 4"),
        _edge_antigravity("task4_brief", "governs", "task4_evaluation_test", TASK4_BRIEF_ANTIGRAVITY, "Task 4"),
        _edge_antigravity("task4_brief", "governs", "task4_scoring", TASK4_BRIEF_ANTIGRAVITY, "Task 4"),
        _edge_antigravity("task4_brief", "governs", "task4_qualification", TASK4_BRIEF_ANTIGRAVITY, "Task 4"),
        _edge_antigravity("task4_brief", "governs", "task4_evaluation", TASK4_BRIEF_ANTIGRAVITY, "Task 4"),
        _edge_antigravity("implementation_plan", "governs", "task4_scoring", PLAN_ANTIGRAVITY, "scoring_antigravity.py"),
        _edge_antigravity("implementation_plan", "governs", "task4_qualification", PLAN_ANTIGRAVITY, "qualification_antigravity.py"),
        _edge_antigravity("implementation_plan", "governs", "task4_evaluation", PLAN_ANTIGRAVITY, "evaluation_antigravity.py"),
        _edge_antigravity("task4_scoring_test", "tests", "task4_scoring", SCORING_TEST_ANTIGRAVITY, "ScoringAntigravityTest"),
        _edge_antigravity("task4_qualification_test", "tests", "task4_qualification", QUALIFICATION_TEST_ANTIGRAVITY, "QualificationAntigravityTest"),
        _edge_antigravity("task4_evaluation_test", "tests", "task4_evaluation", EVALUATION_TEST_ANTIGRAVITY, "EvaluationAntigravityTest"),
        _edge_antigravity("task4_scoring", "calls", "task2_predecessor", SCORING_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("task4_scoring", "calls", "task2_synthetic", SCORING_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("task4_scoring", "calls", "task3_local_stft", SCORING_ANTIGRAVITY, "score_local_stft_antigravity"),
        _edge_antigravity("task4_scoring", "binds", "task3_backend_selection", SCORING_ANTIGRAVITY, "finalize_backend_decision_antigravity"),
        _edge_antigravity("task4_scoring", "binds", "task3_benchmark_artifact", SCORING_ANTIGRAVITY, "exp468_acceleration_benchmark_antigravity.json"),
        _edge_antigravity("task4_qualification", "binds", "task4_scoring", QUALIFICATION_ANTIGRAVITY, "FoldMethodScoresAntigravity"),
        _edge_antigravity("task4_qualification", "calls", "task2_predecessor", QUALIFICATION_ANTIGRAVITY, "score_c0_fold_antigravity"),
        _edge_antigravity("task4_qualification", "calls", "task2_synthetic", QUALIFICATION_ANTIGRAVITY, "primary_rows_antigravity"),
        _edge_antigravity("task4_qualification", "calls", "task3_local_stft", QUALIFICATION_ANTIGRAVITY, "score_local_stft_antigravity"),
        _edge_antigravity("task4_qualification", "calls", "rank_auc", QUALIFICATION_ANTIGRAVITY, "mann_whitney_auc_antigravity"),
        _edge_antigravity("task4_qualification", "binds", "task1_contracts", QUALIFICATION_ANTIGRAVITY, "Exp468QualificationAntigravity"),
        _edge_antigravity("task4_evaluation", "binds", "task1_contracts", EVALUATION_ANTIGRAVITY, "Exp468QualificationAntigravity"),
        _edge_antigravity("task4_report", "informs", "implementation_report", TASK4_REPORT_ANTIGRAVITY, "Task 4 구현 보고서"),
        _edge_antigravity("task4_report", "informs", "dashboard_artifact", TASK4_REPORT_ANTIGRAVITY, "implementation_task_4_complete_no_private_run"),
        _edge_antigravity("implementation_plan", "governs", "task5_artifacts", PLAN_ANTIGRAVITY, "Task 5 — 결과 파일·독립 replay·HTML 대시보드"),
        _edge_antigravity("implementation_plan", "governs", "task5_replay", PLAN_ANTIGRAVITY, "Task 5 — 결과 파일·독립 replay·HTML 대시보드"),
        _edge_antigravity("implementation_plan", "governs", "task5_dashboard_export", PLAN_ANTIGRAVITY, "Task 5 — 결과 파일·독립 replay·HTML 대시보드"),
        _edge_antigravity("task5_artifacts_test", "tests", "task5_artifacts", ARTIFACTS_TEST_ANTIGRAVITY, "ArtifactsAntigravityTest"),
        _edge_antigravity("task5_replay_test", "tests", "task5_replay", REPLAY_TEST_ANTIGRAVITY, "ReplayAntigravityTest"),
        _edge_antigravity("task5_dashboard_export_test", "tests", "task5_dashboard_export", DASHBOARD_EXPORT_TEST_ANTIGRAVITY, "DashboardExportAntigravityTest"),
        _edge_antigravity("task5_artifacts", "calls", "task4_qualification", ARTIFACTS_ANTIGRAVITY, "qualify_antigravity"),
        _edge_antigravity("task5_artifacts", "calls", "task4_evaluation", ARTIFACTS_ANTIGRAVITY, "evaluate_qualification_antigravity"),
        _edge_antigravity("task5_replay", "calls", "task4_qualification", REPLAY_ANTIGRAVITY, "qualify_antigravity"),
        _edge_antigravity("task5_replay", "calls", "task4_evaluation", REPLAY_ANTIGRAVITY, "evaluate_qualification_antigravity"),
        _edge_antigravity("task5_dashboard_export", "consumes", "task5_replay", DASHBOARD_EXPORT_ANTIGRAVITY, "InspectedResultAntigravity"),
        _edge_antigravity("task5_report", "informs", "implementation_report", TASK5_REPORT_ANTIGRAVITY, "Task 5 구현 보고서"),
        _edge_antigravity("task5_report", "informs", "dashboard_artifact", TASK5_REPORT_ANTIGRAVITY, "implementation_task_5_complete_no_private_run"),
        _edge_antigravity("task5_artifacts", "consumed_by", "task6_cli", PLAN_ANTIGRAVITY, "cli_antigravity.py"),
        _edge_antigravity("task5_replay", "consumed_by", "task6_cli", PLAN_ANTIGRAVITY, "inspect --root PATH"),
        _edge_antigravity("task5_artifacts", "consumed_by", "task6_pipeline", PLAN_ANTIGRAVITY, "pipeline"),
        _edge_antigravity("task5_replay", "consumed_by", "task6_pipeline", PLAN_ANTIGRAVITY, "pipeline"),
        _edge_antigravity("task1_config", "documents", "preregistration", PREREGISTRATION_ANTIGRAVITY, "고정한 과학 정의"),
        _edge_antigravity("task1_contracts", "documents", "implementation_report", IMPLEMENTATION_REPORT_ANTIGRAVITY, "qualification은 seed-major 10개 fold"),
        _edge_antigravity("implementation_plan", "documents", "runbook", RUNBOOK_ANTIGRAVITY, "다음 허용 작업"),
        _edge_antigravity("implementation_plan", "documents", "dashboard_artifact", DASHBOARD_ARTIFACT_ANTIGRAVITY, "progress_checklist"),
        _edge_antigravity("implementation_report", "informs", "dashboard_artifact", DASHBOARD_ARTIFACT_ANTIGRAVITY, "exp468_task5_truth"),
        _edge_antigravity("dashboard_artifact", "generates", "dashboard", DASHBOARD_ARTIFACT_ANTIGRAVITY, '"surface": "dashboard"'),
        _edge_antigravity("map_builder", "verifies", "preregistration", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "runbook", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "implementation_report", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task3_fallback_amendment", BUILDER_ANTIGRAVITY, "_validate_fallback_evidence_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task3_fallback_report", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task4_brief", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task4_report", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task5_report", BUILDER_ANTIGRAVITY, "_validated_html_text_antigravity"),
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
            ["task3_brief", "task3_local_stft_test"],
            ["task3_brief", "task3_acceleration_test"],
            ["task3_brief", "task3_local_stft_test", "task3_local_stft"],
            ["task3_brief", "task3_acceleration_test", "task3_acceleration"],
            ["task3_fallback_brief", "task3_backend_selection_test", "task3_backend_selection", "task3_acceleration"],
            ["task3_fallback_brief", "task3_fallback_amendment", "task3_backend_selection"],
            ["task3_benchmark", "task3_acceleration", "task3_local_stft"],
            ["task3_benchmark", "task3_benchmark_artifact", "task3_report", "implementation_report"],
            ["task3_benchmark_artifact", "task3_fallback_report", "implementation_report"],
            ["task4_brief", "task4_scoring_test", "task4_scoring", "task3_backend_selection"],
            ["task4_brief", "task4_qualification_test", "task4_qualification", "task4_scoring"],
            ["task4_qualification", "rank_auc"],
            ["task4_brief", "task4_evaluation_test", "task4_evaluation", "task1_contracts"],
            ["task4_report", "implementation_report", "dashboard_artifact", "dashboard"],
            ["implementation_plan", "preregistration", "runbook", "implementation_report"],
            ["implementation_report", "dashboard_artifact", "dashboard"],
            ["implementation_plan", "task5_artifacts", "task5_artifacts_test", "task5_report"],
            ["implementation_plan", "task5_replay", "task5_replay_test", "task5_report"],
            ["implementation_plan", "task5_dashboard_export", "task5_dashboard_export_test", "task5_report"],
            ["task5_artifacts", "task4_qualification", "task4_evaluation", "task5_replay"],
            ["task5_replay", "task5_dashboard_export", "dashboard"],
            ["task5_artifacts", "task6_cli", "task6_pipeline"],
            ["map_builder", "task4_brief", "task4_report", "implementation_report", "dashboard"],
        ],
        "task1_answers": {
            "callers": "Config is called by contracts/tests and Tasks 2-4; contracts are called by qualification/publication/replay; the plan calls the frozen-doc and dashboard state; the portable generator calls the source-backed artifact; every later preflight calls the map builder.",
            "effects": "Freezes backend-neutral scientific constants and immutable result shapes, preserves the historical implementation_task_1_complete record while the current docs report Task 5 completion, and leaves ECG access, scoring, qualification logic, TEST/FN, routing, and promotion untouched.",
            "tests": "test_config_antigravity and test_contracts_antigravity cover executable behavior; build_codemap_exp468_antigravity.py --validate-only covers evidence, required edges, HTML/JSON structure, embedded JSON, source-backed dashboard truth, and tracked fingerprints.",
        },
        "task2_answers": {
            "callers": "The four focused Task 2 tests call the Exp468 adapters now; future Task 3-4 scoring and qualification call the loader/interpolator/synthetic/C0 boundaries. Each Exp468 adapter directly imports its corresponding reviewed Exp466 authority.",
            "effects": "Preserves the exact Exp466 function objects and therefore the approved TRAIN MD5 and 69x96 boundary, 96-to-140 interpolation, four synthetic scopes including period_change, and C0 normal/primary dtype, shape, order, and bytes. It adds no loading, interpolation, synthetic, or scoring algorithm and leaves TEST/FN, qualification, routing, official results, and promotion untouched.",
            "tests": "test_sources_antigravity covers object identity, approved TRAIN MD5/69x96, and pre-open TEST-name rejection; test_interpolation_antigravity covers exact object identity; test_synthetic_antigravity covers exact identity and four-scope order; test_predecessor_antigravity covers both exact C0 objects and fixed-fold normal/four-scope byte parity. build_codemap_exp468_antigravity.py --validate-only covers the exact-import edges, dashboard truth, HTML/JSON structure, embedded JSON, and tracked fingerprints.",
        },
        "task3_answers": {
            "callers": "The original Task 3 brief, plan, and focused suites call local_stft and acceleration; acceleration calls local_stft; the TRAIN-only benchmark calls acceleration. The fallback brief and approval amendment govern backend_selection, whose behavior suite and Task 4 scorer read the approval-linked benchmark artifact. Task 4 scoring and raw replay call the selected NumPy scorer; reports, dashboard, map, and controller consume the same evidence.",
            "effects": "Preserves the frozen S10 science, all acceleration measurements, and gate_passed=false while exposing a separate immutable operational decision with fallback_authorized=true and selected_backend=numpy to Task 4. Approval document SHA, pre-approval benchmark bytes SHA, every non-NumPy failure reason, exact 552/690/1242/6210 counts, parity, versions/device, zero runs, and null results/hashes remain auditable.",
            "tests": "test_local_stft_antigravity covers the frozen score science; test_acceleration_antigravity covers complete NumPy/Numba parity, deterministic gate truth, counts, and validation; test_backend_selection_antigravity covers gate-success invariants, approval-required fallback, candidate failure reasons, lowercase SHA-256 evidence, immutability, and the real approval artifact link. The map validator covers exact benchmark preservation, approval hashes, dashboard truth, HTML structure, edges, and fingerprints.",
        },
        "task4_answers": {
            "callers": "The Task 4 brief and checked plan govern the three production modules and their test-first suites. scoring consumes the exact Task 2 C0/synthetic adapters, Task 3 local scorer, checked benchmark artifact, and immutable approved NumPy decision. qualification consumes immutable fold records and independently calls C0, synthetic, NumPy local-STFT, exact contracts, and reviewed Mann-Whitney AUC. evaluation consumes only scalar qualification contracts. Task 4 reports, dashboard, map, controller, and Task 5 publication/replay consume these outputs.",
            "effects": "Adds no-selector C0/S10/C10 assembly with one shared S10 calibration, owned read-only raw evidence and input, exact five blocked folds across two seeds, identical TRAIN-byte and OOF enforcement, raw-input independent exact replay, same-fold per-type AUC, every frozen per-fold gate in deterministic order, 10-of-10 and per-seed zero-normal scalar aggregation, and only awaiting-private or blocked-before-private status. It adds no mean gate, result root, CLI, publication, artifact hash, private/official run, TEST/FN read, routing, champion/pending change, or promotion.",
            "tests": "test_scoring_antigravity covers the approved SHA-bound NumPy path, absent backend selector, combined-query calibration, exact C0/S10/C10 and strict fallback, validation, ownership, and immutability. test_qualification_antigravity covers exact fold arrays/order/coverage/TRAIN bytes, independent C0/raw/rank/confirmation/C10 replay tamper rejection, same-fold AUC, frozen boundary/reason order, ten rows, and absence of a mean gate. test_evaluation_antigravity covers scalar-only 10-of-10/per-seed aggregation, exact boundary statuses, immutable decision, and zero external effects. The map validator covers Task 4 nodes/edges/evidence, dashboard truth, HTML structure, embedded JSON, and fingerprints.",
        },
        "task5_answers": {
            "callers": "Task 5 publication tests call publish_result_antigravity, replay tests call inspect_result_antigravity, and the dashboard suite calls export_dashboard_antigravity. Publication and replay consume only the Task 4 FoldMethodScoresAntigravity, Exp468QualificationAntigravity, and QualificationDecisionAntigravity contracts; future Task 6 CLI/pipeline consumes the seven-leaf root and source-free inspect boundary.",
            "effects": "Writes exactly seven canonical leaves: one raw-evidence context, one dashboard snapshot, one scalar decision, three deterministic JSONL projections, and a self-excluding six-leaf SHA-256 map. It preserves 69x140 little-endian float64 C-order inputs, fit/query indices, raw JS/rank/confirmation/C10/AUC evidence, backend numpy, approval and pre-approval benchmark SHA values, and zero private/official execution truth. Replay recomputes Task 4 from retained inputs and rejects semantic tampering even after outer hashes are repaired; HTML remains deterministic and self-contained.",
            "tests": "test_artifacts_antigravity covers exact seven-leaf/self-excluding hash contents, canonical JSON/newline bytes, empty failed-fold output, seed-to-fold-to-method 30-row order, C0/S10/C10 summary order, provenance fields, and pre-write rejection of missing/additional/reordered records, fold swaps, wrong blocked splits, and forged qualification/decision without creating a root. test_replay_antigravity covers loader-free exact replay, every retained raw/metric/provenance/dashboard/decision/context/summary/failed-fold mutation, stale/repaired hash rejection, duplicate keys, malformed base64/shape/digest/NaN, and extra leaves. test_dashboard_export_antigravity covers deterministic Korean HTML with per-fold period evidence, C0 improvement, normal confirmation errors, qualification PASS/FAIL private status, both private runs not_run, and checked Task 5 items. The map validator covers Task 5 nodes, Task 6 consumer flows, HTML/dashboard truth, and new fingerprints.",
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
    answers = graph["task5_answers"]
    embedded = html_module.escape(
        json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp468 Task 5 publication 코드맵</title>
<style>body{{max-width:1240px;margin:2rem auto;padding:0 1rem;font:15px/1.65 system-ui;color:#172033}}table{{width:100%;border-collapse:collapse;margin:1rem 0}}th,td{{border:1px solid #aab3bf;padding:.48rem;text-align:left;vertical-align:top}}code,pre{{background:#eef2f7;padding:.1rem .3rem;overflow:auto}}.box{{background:#f5f7fa;border-left:6px solid #2457a6;padding:1rem}}</style></head>
<body><main><h1>Exp468 Task 5 publication 코드맵</h1>
<div class="box"><p><strong>상태:</strong> <code>implementation_task_5_complete_no_private_run</code>. 이 지도는 Task 1-5 경계, 정확한 일곱 leaf publication, raw-input 독립 replay, 결정적 HTML dashboard, Task 6 CLI/pipeline 소비 edge와 호출자·영향·검사를 기록한다.</p></div>
<h2>세 질문</h2><table><tr><th>무엇이 호출하는가?</th><td>{html_module.escape(answers['callers'])}</td></tr><tr><th>무엇에 영향을 주는가?</th><td>{html_module.escape(answers['effects'])}</td></tr><tr><th>어떤 검사가 덮는가?</th><td>{html_module.escape(answers['tests'])}</td></tr></table>
<h2>노드</h2><table><thead><tr><th>ID</th><th>파일</th><th>호출자와 영향</th><th>검사</th></tr></thead><tbody>{node_rows}</tbody></table>
<h2>관계</h2><table><thead><tr><th>시작</th><th>관계</th><th>도착</th><th>근거</th></tr></thead><tbody>{edge_rows}</tbody></table>
<h2>기계 판독 JSON</h2><pre id="codemap-json-antigravity">{embedded}</pre>
</main></body></html>"""


def _validate_fallback_evidence_antigravity() -> None:
    payload = json.loads(
        (ROOT_ANTIGRAVITY / BENCHMARK_ARTIFACT_ANTIGRAVITY).read_text(
            encoding="utf-8",
        ),
    )
    expected_truth = {
        "status": "TASK_3_COMPLETE_NUMPY_FALLBACK_AUTHORIZED",
        "gate_passed": False,
        "fallback_authorized": True,
        "selected_backend": "numpy",
        "official_runs": 0,
        "private_runs": 0,
        "result": None,
        "result_hash": None,
    }
    for key, expected in expected_truth.items():
        if payload.get(key) != expected:
            raise ValueError(f"benchmark fallback truth differs: {key}")
    authorization = payload.get("fallback_authorization")
    if not isinstance(authorization, dict) or authorization.get("authorized") is not True:
        raise ValueError("benchmark fallback authorization is missing")
    if authorization.get("approval_date") != "2026-08-24":
        raise ValueError("benchmark approval date differs")
    if authorization.get("approval_document_path") != FALLBACK_AMENDMENT_ANTIGRAVITY:
        raise ValueError("benchmark approval document path differs")
    if (
        authorization.get("pre_approval_benchmark_sha256")
        != PRE_APPROVAL_BENCHMARK_SHA256_ANTIGRAVITY
    ):
        raise ValueError("pre-approval benchmark SHA-256 differs")
    approval_bytes = (ROOT_ANTIGRAVITY / FALLBACK_AMENDMENT_ANTIGRAVITY).read_bytes()
    if authorization.get("approval_document_sha256") != hashlib.sha256(
        approval_bytes,
    ).hexdigest():
        raise ValueError("approval document SHA-256 differs")

    numpy_backend = payload["backends"]["numpy"]
    numba_backend = payload["backends"]["numba"]
    mlx_backend = payload["backends"]["mlx"]
    exact_measurements = (
        (numpy_backend, "warm_median_seconds", 0.03191525023430586),
        (numpy_backend, "warm_iqr_seconds", 0.0004911664873361588),
        (numpy_backend, "gate_seconds", 0.03191525023430586),
        (numpy_backend, "speedup", 1.0),
        (numba_backend, "backend_preparation_seconds", 0.10018991678953171),
        (numba_backend, "cold_compile_seconds", 0.44130249973386526),
        (numba_backend, "cold_total_seconds", 0.5741005414165556),
        (numba_backend, "warm_median_seconds", 0.032608124893158674),
        (numba_backend, "warm_iqr_seconds", 0.0014127083122730255),
        (numba_backend, "gate_seconds", 0.5741005414165556),
        (numba_backend, "speedup", 0.05559174383559552),
        (numba_backend, "max_raw_abs_error", 5.551115123125783e-16),
    )
    for backend, key, expected in exact_measurements:
        if backend.get(key) != expected:
            raise ValueError(f"benchmark measurement differs: {key}")
    initial = payload["profile_and_batching_review"]["initial_measurement"]
    expected_initial = {
        "numba_cold_gate_seconds": 1.2641031672246754,
        "numba_max_raw_abs_error": 5.967448757360216e-16,
        "numba_speedup": 0.02543374318247438,
        "numba_warm_median_seconds": 0.029086542315781116,
        "numpy_median_seconds": 0.03215087531134486,
    }
    if initial != expected_initial:
        raise ValueError("initial benchmark measurements differ")
    if payload["timing_protocol"].get("required_speedup") != 1.1:
        raise ValueError("required acceleration speedup differs")
    if not isinstance(payload.get("workload"), dict):
        raise ValueError("benchmark workload must be a mapping")
    for key, expected in {
        "fit_raw": 552,
        "query_raw": 690,
        "transformed_rows": 1242,
        "frames": 6210,
        "length_28_frame_transforms": 6210,
    }.items():
        if payload["workload"].get(key) != expected:
            raise ValueError(f"benchmark workload differs: {key}")
    mlx_reason = mlx_backend.get("fallback_reason")
    if not isinstance(mlx_reason, str) or "float64 is not supported on the GPU" not in mlx_reason:
        raise ValueError("MLX float64 failure reason differs")


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
        "status": "implementation_task_5_complete_no_private_run",
        "acceleration_gate_passed": False,
        "fallback_authorized": True,
        "selected_backend": "numpy",
        "focused_tests": 12,
        "exp468_tests": 66,
        "exp466_regression_tests": 56,
        "score_calculations": 2,
        "score_calculation_scope": "acceleration_benchmark_full_workload_initial_and_improved",
        "publication_leaves": 7,
        "metric_rows": 30,
        "summary_rows": 3,
        "official_runs": 0,
        "private_runs": 0,
        "official_result": None,
        "private_result": None,
        "result_hash": None,
    }]
    if summary != expected:
        raise ValueError("dashboard Task 4 implementation truth differs")
    source_ids = {item.get("id") for item in artifact["sources"]}
    if "exp468_task5_truth" not in source_ids:
        raise ValueError("dashboard Task 5 source is missing")
    rows = artifact["snapshot"]["datasets"].get("progress_checklist", [])
    if artifact.get("snapshot", {}).get("status") != "ready":
        raise ValueError("dashboard snapshot status differs")
    if not any(
        row.get("task") == "Task 4 C0/S10/C10·exact 10-fold gate"
        and row.get("checked") == "☑"
        and row.get("status") == "완료 — private 실행 없음"
        for row in rows
    ):
        raise ValueError("dashboard Task 4 completion checklist truth differs")
    rendered = _validated_html_text_antigravity(ROOT_ANTIGRAVITY / DASHBOARD_ANTIGRAVITY)
    for required in (
        "implementation_task_5_complete_no_private_run",
        "exp468_task5_truth",
        "Task 4 C0/S10/C10·exact 10-fold gate",
        "Task 5 publication·artifact replay·dashboard",
        "focused 12/12",
        "period 변화",
        "C0보다 개선한 fold",
        "정상 확인 오류",
        "private 재현성 확인 대기",
        "아직 실행하지 않음",
        "fallback_authorized=true",
        "gate_passed=false",
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
    if set(graph.get("task3_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 3 caller/effect/test answers are incomplete")
    if set(graph.get("task4_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 4 caller/effect/test answers are incomplete")
    if set(graph.get("task5_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 5 caller/effect/test answers are incomplete")
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
        "task1_brief", "task2_brief", "task3_brief", "task3_fallback_brief", "task4_brief", "approved_design", "implementation_plan", "task1_config", "task1_contracts",
        "task1_config_test", "task1_contracts_test", "preregistration", "runbook",
        "implementation_report", "dashboard_artifact", "dashboard", "map_builder",
        "task2_sources", "task2_interpolation", "task2_synthetic", "task2_predecessor",
        "task2_sources_test", "task2_interpolation_test", "task2_synthetic_test",
        "task2_predecessor_test", "exp466_sources", "exp466_interpolation",
        "exp466_synthetic", "exp466_predecessor",
        "task3_local_stft_test", "task3_acceleration_test", "task3_backend_selection_test",
        "task3_local_stft", "task3_acceleration", "task3_backend_selection", "task3_benchmark",
        "task3_benchmark_artifact", "task3_report", "task3_fallback_amendment",
        "task3_fallback_report",
        "task4_scoring_test", "task4_qualification_test", "task4_evaluation_test",
        "task4_scoring", "task4_qualification", "task4_evaluation", "task4_report",
        "task5_artifacts", "task5_replay", "task5_dashboard_export",
        "task5_artifacts_test", "task5_replay_test", "task5_dashboard_export_test",
        "task5_report", "task6_cli", "task6_pipeline", "rank_auc",
    }
    if ids != required_nodes:
        raise ValueError("Task 1-5 nodes differ")
    allowed_types = {"governs", "tests", "imports_exact", "documents", "informs", "generates", "verifies", "calls", "binds", "authorizes", "consumes", "consumed_by"}
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
        ("task3_brief", "governs", "task3_local_stft_test"),
        ("task3_brief", "governs", "task3_acceleration_test"),
        ("task3_fallback_brief", "governs", "task3_backend_selection_test"),
        ("task3_fallback_brief", "governs", "task3_backend_selection"),
        ("task3_fallback_brief", "governs", "task3_fallback_amendment"),
        ("task3_fallback_brief", "governs", "task3_benchmark_artifact"),
        ("implementation_plan", "governs", "task3_local_stft_test"),
        ("implementation_plan", "governs", "task3_acceleration_test"),
        ("task3_brief", "governs", "task3_local_stft"),
        ("task3_brief", "governs", "task3_acceleration"),
        ("implementation_plan", "governs", "task3_local_stft"),
        ("implementation_plan", "governs", "task3_acceleration"),
        ("task1_config", "governs", "task3_local_stft"),
        ("task3_local_stft_test", "tests", "task3_local_stft"),
        ("task3_acceleration_test", "tests", "task3_acceleration"),
        ("task3_backend_selection_test", "tests", "task3_backend_selection"),
        ("task3_backend_selection_test", "tests", "task3_benchmark_artifact"),
        ("task3_acceleration", "calls", "task3_local_stft"),
        ("task3_backend_selection", "binds", "task3_acceleration"),
        ("task3_fallback_amendment", "authorizes", "task3_backend_selection"),
        ("task3_benchmark", "calls", "task3_acceleration"),
        ("task3_benchmark", "generates", "task3_benchmark_artifact"),
        ("task3_benchmark_artifact", "informs", "task3_report"),
        ("task3_report", "informs", "implementation_report"),
        ("task3_benchmark_artifact", "informs", "task3_fallback_report"),
        ("task3_fallback_report", "informs", "implementation_report"),
        ("task3_benchmark_artifact", "informs", "dashboard_artifact"),
        ("task4_brief", "governs", "task4_scoring_test"),
        ("task4_brief", "governs", "task4_qualification_test"),
        ("task4_brief", "governs", "task4_evaluation_test"),
        ("task4_brief", "governs", "task4_scoring"),
        ("task4_brief", "governs", "task4_qualification"),
        ("task4_brief", "governs", "task4_evaluation"),
        ("implementation_plan", "governs", "task4_scoring"),
        ("implementation_plan", "governs", "task4_qualification"),
        ("implementation_plan", "governs", "task4_evaluation"),
        ("task4_scoring_test", "tests", "task4_scoring"),
        ("task4_qualification_test", "tests", "task4_qualification"),
        ("task4_evaluation_test", "tests", "task4_evaluation"),
        ("task4_scoring", "calls", "task2_predecessor"),
        ("task4_scoring", "calls", "task2_synthetic"),
        ("task4_scoring", "calls", "task3_local_stft"),
        ("task4_scoring", "binds", "task3_backend_selection"),
        ("task4_scoring", "binds", "task3_benchmark_artifact"),
        ("task4_qualification", "binds", "task4_scoring"),
        ("task4_qualification", "calls", "task2_predecessor"),
        ("task4_qualification", "calls", "task2_synthetic"),
        ("task4_qualification", "calls", "task3_local_stft"),
        ("task4_qualification", "calls", "rank_auc"),
        ("task4_qualification", "binds", "task1_contracts"),
        ("task4_evaluation", "binds", "task1_contracts"),
        ("task4_report", "informs", "implementation_report"),
        ("task4_report", "informs", "dashboard_artifact"),
        ("implementation_plan", "governs", "task5_artifacts"),
        ("implementation_plan", "governs", "task5_replay"),
        ("implementation_plan", "governs", "task5_dashboard_export"),
        ("task5_artifacts_test", "tests", "task5_artifacts"),
        ("task5_replay_test", "tests", "task5_replay"),
        ("task5_dashboard_export_test", "tests", "task5_dashboard_export"),
        ("task5_artifacts", "calls", "task4_qualification"),
        ("task5_artifacts", "calls", "task4_evaluation"),
        ("task5_replay", "calls", "task4_qualification"),
        ("task5_replay", "calls", "task4_evaluation"),
        ("task5_dashboard_export", "consumes", "task5_replay"),
        ("task5_report", "informs", "implementation_report"),
        ("task5_report", "informs", "dashboard_artifact"),
        ("task5_artifacts", "consumed_by", "task6_cli"),
        ("task5_replay", "consumed_by", "task6_cli"),
        ("task5_artifacts", "consumed_by", "task6_pipeline"),
        ("task5_replay", "consumed_by", "task6_pipeline"),
        ("implementation_plan", "documents", "dashboard_artifact"),
        ("implementation_report", "informs", "dashboard_artifact"),
        ("dashboard_artifact", "generates", "dashboard"),
        ("map_builder", "verifies", "preregistration"),
        ("map_builder", "verifies", "runbook"),
        ("map_builder", "verifies", "implementation_report"),
        ("map_builder", "verifies", "task3_fallback_amendment"),
        ("map_builder", "verifies", "task3_fallback_report"),
        ("map_builder", "verifies", "task4_brief"),
        ("map_builder", "verifies", "task4_report"),
        ("map_builder", "verifies", "dashboard_artifact"),
        ("map_builder", "verifies", "dashboard"),
    }
    if not required_edges.issubset(edge_keys):
        raise ValueError("Task 1-5 required edges are incomplete")
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
        TASK3_BRIEF_ANTIGRAVITY,
        TASK3_FALLBACK_BRIEF_ANTIGRAVITY,
        TASK4_BRIEF_ANTIGRAVITY,
        TASK3_REPORT_ANTIGRAVITY,
        TASK3_FALLBACK_REPORT_ANTIGRAVITY,
        TASK4_REPORT_ANTIGRAVITY,
        TASK5_REPORT_ANTIGRAVITY,
        FALLBACK_AMENDMENT_ANTIGRAVITY,
        PREREGISTRATION_ANTIGRAVITY,
        RUNBOOK_ANTIGRAVITY,
        IMPLEMENTATION_REPORT_ANTIGRAVITY,
    ):
        _validated_html_text_antigravity(ROOT_ANTIGRAVITY / relative)
    _validate_fallback_evidence_antigravity()
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
