#!/usr/bin/env python3
"""Build and validate the bounded Exp466 Task 1 code-map triplet."""

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
    BUILDER_ANTIGRAVITY,
    "docs/codemap/codemap_exp465_antigravity.html",
    "docs/codemap/codemap_exp465_antigravity.json",
    "docs/codemap/codemap_exp465_antigravity.lock",
)
GRAPH_SCHEMA_ANTIGRAVITY = "exp466_task1_authority_contract_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp466_task1_authority_contract_codemap_lock_v1"
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
    map_test = (BUILDER_ANTIGRAVITY,)
    config_test = (CONFIG_TEST_ANTIGRAVITY,)
    contracts_test = (CONTRACTS_TEST_ANTIGRAVITY,)
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
            "preregistration",
            "docs/EXP466_PREREGISTRATION_antigravity.html",
            "Caller: governance and scientific reviewers. Effect: states planned_implementation_not_started, freezes C0/P9/C9 and C9 gates, records zero/null truth, CPU/MLX pre-run gate, and forbidden scope.",
            "planned_implementation_not_started",
            map_test,
        ),
        _node_antigravity(
            "runbook",
            "docs/EXP466_RUNBOOK_antigravity.html",
            "Caller: future operators. Effect: keeps official root absent, lists focused read-only verification, preserves TEST prohibition and zero/null counts, and defers all score/private/official commands.",
            "Task 1 focused verification",
            map_test,
        ),
        _node_antigravity(
            "implementation_report",
            "docs/EXP466_IMPLEMENTATION_REPORT_antigravity.html",
            "Caller: Task 1 closeout and future audit. Effect: records exact RED, two GREEN runs, map/diff verification, no-workload MLX decision, planned status, and preserved boundaries without qualification claims.",
            "TDD 증거",
            map_test,
        ),
        _node_antigravity(
            "dashboard",
            "docs/EXP466_DASHBOARD_antigravity.html",
            "Caller: user-facing progress review. Effect: preserves prior approval checks, marks only Task 1 scaffold complete, reports planned status, zero/null execution truth, and unchanged prohibitions.",
            "planned_implementation_not_started",
            map_test,
        ),
        _node_antigravity(
            "research_state",
            "RESEARCH_STATE.json",
            "Caller: research-state audit. Effect: adds one Exp466 planned entry with frozen science contract, zero official/private/label activity, null hashes, and no routing/promotion/pending changes.",
            '"exp466"',
            map_test,
        ),
        _node_antigravity(
            "experiment_catalog",
            "EXPERIMENT_CATALOG.json",
            "Caller: experiment inventory. Effect: adds one Exp466 planned entry with package/documents/authority, exact scientific contract, zero execution counts, and null result/hash fields.",
            '"id": "Exp466"',
            map_test,
        ),
        _node_antigravity(
            "champion_ledger",
            "CHAMPION_LEDGER.json",
            "Caller: champion-history audit. Effect: records Exp466 planning while leaving champion and pending candidate unchanged and disallowing routing/promotion/result claims.",
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
            "map_builder",
            BUILDER_ANTIGRAVITY,
            "Caller: Task 1 closeout verification. Effect: generates canonical HTML/JSON/lock or validates evidence, HTML structure, graph relations, embedded JSON, Exp465 baseline fingerprints, and all tracked fingerprints without writing in validate-only mode.",
            "build_triplet_antigravity",
            map_test,
        ),
    ]
    edges = [
        _edge_antigravity("task1_brief", "governs", "task1_config", BRIEF_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("approved_design", "governs", "task1_config", DESIGN_ANTIGRAVITY, "고정 입력"),
        _edge_antigravity("exp465_baseline", "informs_structure", "map_builder", "docs/codemap/codemap_exp465_antigravity.json", '"map_builder"'),
        _edge_antigravity("task1_config", "governs", "task1_contracts", CONFIG_ANTIGRAVITY, "FROZEN_CONFIG_ANTIGRAVITY"),
        _edge_antigravity("task1_config_test", "tests", "task1_config", CONFIG_TEST_ANTIGRAVITY, "test_frozen_config_antigravity"),
        _edge_antigravity("task1_contracts_test", "tests", "task1_contracts", CONTRACTS_TEST_ANTIGRAVITY, "test_qualification_rejects_mutable_reordered_and_wrong_69_counts_antigravity"),
        _edge_antigravity("map_builder", "verifies", "preregistration", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "runbook", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "implementation_report", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "dashboard", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "research_state", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "experiment_catalog", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "champion_ledger", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
        _edge_antigravity("map_builder", "verifies", "task1_report", BUILDER_ANTIGRAVITY, "validate_map_antigravity"),
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
        "flows": [
            "task1 brief + approved design -> frozen config -> immutable contracts -> future Tasks 2-6",
            "focused config/contracts RED-GREEN tests -> executable Task 1 authority verification",
            "truth HTML + additive planned ledgers + dashboard -> map builder -> canonical triplet and read-only fingerprint validation",
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
        '<title>Exp466 Task 1 코드맵</title><style>body{max-width:1180px;margin:2rem auto;padding:0 1rem;font:16px/1.7 system-ui;color:#172033}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}pre{padding:1rem}</style></head><body><main>'
        '<h1>Exp466 Task 1 코드맵</h1><p>이 map은 Exp465 triplet을 구조 기준으로 삼아 Exp466의 caller, effect, test 관계를 고정한다. Task 1은 권한·불변 계약·planned zero/null truth만 다루며 ECG200 로딩, 점수 계산, private/official 실행은 포함하지 않는다.</p>'
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
    if set(graph.get("task1_answers", {})) != {"callers", "effects", "tests"}:
        raise ValueError("Task 1 caller/effect/test answers are incomplete")
    if lock.get("source_parent_commit") != graph.get("source_parent_commit"):
        raise ValueError("graph/lock source mismatch")
    if lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise ValueError("lock represented paths differ")
    if lock.get("working_tree_dirty") is not False:
        raise ValueError("lock must record a closed generation boundary")
    if lock.get("fingerprint_count") != len(TRACKED_PATHS_ANTIGRAVITY):
        raise ValueError("lock fingerprint count differs")
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
