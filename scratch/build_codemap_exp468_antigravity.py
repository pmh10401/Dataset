#!/usr/bin/env python3
"""Build and validate the bounded Exp468 Task 1 code-map triplet."""

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
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity-design.html"
)
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html"
)
BRIEF_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-23-exp468-ecg200-local-stft-spectral-redistribution-antigravity.html/"
    "task-1-brief-antigravity.html"
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
    BRIEF_ANTIGRAVITY,
    CONFIG_ANTIGRAVITY,
    CONTRACTS_ANTIGRAVITY,
    CONFIG_TEST_ANTIGRAVITY,
    CONTRACTS_TEST_ANTIGRAVITY,
    PREREGISTRATION_ANTIGRAVITY,
    RUNBOOK_ANTIGRAVITY,
    IMPLEMENTATION_REPORT_ANTIGRAVITY,
    DASHBOARD_ARTIFACT_ANTIGRAVITY,
    DASHBOARD_ANTIGRAVITY,
    BUILDER_ANTIGRAVITY,
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp468_task1_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp468_task1_codemap_lock_v1"
SOURCE_PARENT_COMMIT_ANTIGRAVITY = "a7dd2430c9af9a82506aa8287447506bb5e7ac5a"
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
            BRIEF_ANTIGRAVITY,
            "Caller: exact Task 1 implementation authority. Effect: fixes file ownership, frozen values including change_types, required interfaces, RED/GREEN sequence, document/dashboard/map outputs, validation commands, report contract, and the no-data/no-score boundary.",
            "Read this first — it is your requirements",
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
            "Caller: Task-by-Task controller and later implementers. Effect: records Task 1 checked completion, exact change_types order, backend-neutral Hann definition, file ownership, and the remaining Task 2-6 boundaries.",
            "implementation_task_1_complete",
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
            "Caller: Task 1 closeout and later audit. Effect: records RED/GREEN scope, executable interfaces, the Task 2/4 responsibility split, backend neutrality, and zero/null pre-run truth.",
            "Task 1 완료 범위",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "dashboard_artifact",
            DASHBOARD_ARTIFACT_ANTIGRAVITY,
            "Caller: portable dashboard generator. Effect: stores bounded source-backed Task 1 status, official/score counts, null result truth, checklist state, and unchanged Exp466 comparison evidence.",
            "implementation_summary",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "dashboard",
            DASHBOARD_ANTIGRAVITY,
            "Caller: user-facing progress review. Effect: renders the source-backed Task 1 completion truth and points only to Task 2 exact-import work while preserving zero official/score execution and no TEST/FN claim.",
            "implementation_task_1_complete",
            [BUILDER_ANTIGRAVITY],
        ),
        _node_antigravity(
            "map_builder",
            BUILDER_ANTIGRAVITY,
            "Caller: Task 1 closeout and every later Exp468 code-changing task preflight. Effect: generates the canonical HTML/JSON/lock triplet or validates structure, evidence, required edges, embedded JSON, dashboard truth, and all tracked fingerprints without writing in validate-only mode.",
            "build_triplet_antigravity",
            [BUILDER_ANTIGRAVITY],
        ),
    ]
    edges = [
        _edge_antigravity("task1_brief", "governs", "task1_config", BRIEF_ANTIGRAVITY, "고정 설정"),
        _edge_antigravity("task1_brief", "governs", "task1_contracts", BRIEF_ANTIGRAVITY, "필수 인터페이스"),
        _edge_antigravity("approved_design", "governs", "task1_config", DESIGN_ANTIGRAVITY, "고정 입력과 범위"),
        _edge_antigravity("implementation_plan", "governs", "task1_config", PLAN_ANTIGRAVITY, "고정 설정·결과 계약·문서 골격"),
        _edge_antigravity("implementation_plan", "governs", "task1_contracts", PLAN_ANTIGRAVITY, "Exp468QualificationAntigravity"),
        _edge_antigravity("task1_config", "governs", "task1_contracts", CONFIG_ANTIGRAVITY, "change_types"),
        _edge_antigravity("task1_config_test", "tests", "task1_config", CONFIG_TEST_ANTIGRAVITY, "test_frozen_stft_authority_antigravity"),
        _edge_antigravity("task1_contracts_test", "tests", "task1_contracts", CONTRACTS_TEST_ANTIGRAVITY, "test_fold_auc_tuples_are_exact_finite_and_positionally_bound_antigravity"),
        _edge_antigravity("task1_config", "documents", "preregistration", PREREGISTRATION_ANTIGRAVITY, "고정한 과학 정의"),
        _edge_antigravity("task1_contracts", "documents", "implementation_report", IMPLEMENTATION_REPORT_ANTIGRAVITY, "qualification은 seed-major 10개 fold"),
        _edge_antigravity("implementation_plan", "documents", "runbook", RUNBOOK_ANTIGRAVITY, "다음 허용 작업"),
        _edge_antigravity("implementation_plan", "documents", "dashboard_artifact", DASHBOARD_ARTIFACT_ANTIGRAVITY, "progress_checklist"),
        _edge_antigravity("implementation_report", "informs", "dashboard_artifact", DASHBOARD_ARTIFACT_ANTIGRAVITY, "exp468_task1_truth"),
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
            ["implementation_plan", "preregistration", "runbook", "implementation_report"],
            ["implementation_report", "dashboard_artifact", "dashboard"],
            ["map_builder", "preregistration", "runbook", "implementation_report", "dashboard"],
        ],
        "task1_answers": {
            "callers": "Config is called by contracts/tests and future Tasks 2-4; contracts are called by future qualification/publication/replay; the plan calls Task 1 docs and dashboard state; the portable generator calls the source-backed artifact; Task 1 and later preflights call the map builder.",
            "effects": "Freezes backend-neutral scientific constants and immutable result shapes, records implementation_task_1_complete with zero score/private/official/label activity and null results/hashes, and leaves ECG access, scoring, qualification logic, TEST/FN, routing, and promotion untouched.",
            "tests": "test_config_antigravity and test_contracts_antigravity cover executable behavior; build_codemap_exp468_antigravity.py --validate-only covers evidence, required edges, HTML/JSON structure, embedded JSON, source-backed dashboard truth, and tracked fingerprints.",
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
    answers = graph["task1_answers"]
    embedded = html_module.escape(
        json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp468 Task 1 코드맵</title>
<style>body{{max-width:1240px;margin:2rem auto;padding:0 1rem;font:15px/1.65 system-ui;color:#172033}}table{{width:100%;border-collapse:collapse;margin:1rem 0}}th,td{{border:1px solid #aab3bf;padding:.48rem;text-align:left;vertical-align:top}}code,pre{{background:#eef2f7;padding:.1rem .3rem;overflow:auto}}.box{{background:#f5f7fa;border-left:6px solid #2457a6;padding:1rem}}</style></head>
<body><main><h1>Exp468 Task 1 코드맵</h1>
<div class="box"><p><strong>상태:</strong> <code>implementation_task_1_complete</code>. 이 지도는 설정·계약·검사·계획·truth 문서·source-backed 대시보드의 현재 경계를 기록한다.</p></div>
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
        "status": "implementation_task_1_complete",
        "score_calculations": 0,
        "official_runs": 0,
        "official_result": None,
        "private_result": None,
        "result_hash": None,
    }]
    if summary != expected:
        raise ValueError("dashboard Task 1 implementation truth differs")
    source_ids = {item.get("id") for item in artifact["sources"]}
    if "exp468_task1_truth" not in source_ids:
        raise ValueError("dashboard Task 1 source is missing")
    rows = artifact["snapshot"]["datasets"].get("progress_checklist", [])
    if not any(
        row.get("task") == "Task 1 고정 설정·결과 계약·문서 골격"
        and row.get("checked") == "☑"
        and row.get("status") == "완료"
        for row in rows
    ):
        raise ValueError("dashboard Task 1 checklist is not complete")
    rendered = _validated_html_text_antigravity(ROOT_ANTIGRAVITY / DASHBOARD_ANTIGRAVITY)
    for required in (
        "implementation_task_1_complete",
        "exp468_task1_truth",
        "Task 1 고정 설정·결과 계약·문서 골격",
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
        "task1_brief", "approved_design", "implementation_plan", "task1_config", "task1_contracts",
        "task1_config_test", "task1_contracts_test", "preregistration", "runbook",
        "implementation_report", "dashboard_artifact", "dashboard", "map_builder",
    }
    if ids != required_nodes:
        raise ValueError("Task 1 nodes differ")
    allowed_types = {"governs", "tests", "documents", "informs", "generates", "verifies"}
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
        raise ValueError("Task 1 required edges are incomplete")
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
        BRIEF_ANTIGRAVITY,
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
