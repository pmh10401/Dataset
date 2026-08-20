"""Build and validate the bounded Exp463 Task 1 code-map triplet."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp463_antigravity.html",
    "docs/codemap/codemap_exp463_antigravity.json",
    "docs/codemap/codemap_exp463_antigravity.lock",
)
SOURCE_PARENT_COMMIT_ANTIGRAVITY = "135c97313d9e107626e6b051e0416e4984e74a08"
GENERATED_AT_ANTIGRAVITY = "2026-08-21T00:00:00Z"
GRAPH_SCHEMA_ANTIGRAVITY = "exp463_task1_governance_codemap_v1"
LOCK_SCHEMA_ANTIGRAVITY = "exp463_task1_governance_codemap_lock_v1"


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


def _build_graph_antigravity() -> dict[str, Any]:
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
            "Caller: Task 1–6 execution. Effect: records bounded files, frozen M0–M6 IDs, private gate stop, and unchecked future task controls.",
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
            "Caller: Task 1 handoff. Effect: records RED/GREEN evidence, validation, commit handoff, and review-pending concern.",
            symbol="TDD evidence",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "dashboard",
            "docs/EXP463_DASHBOARD_antigravity.html",
            "Caller: status review. Effect: shows planned state, M2 primary, easy Korean explanation, aggregate gates, official0, label0, and null artifact map.",
            symbol="Exp463 진행 대시보드",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "research_state",
            "RESEARCH_STATE.json",
            "Caller: governance ledger. Effect: records one planned Exp463 entry with null result hashes, official0, label0, absent root, and unchanged pending state.",
            symbol="\"exp463\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "experiment_catalog",
            "EXPERIMENT_CATALOG.json",
            "Caller: experiment catalog. Effect: registers the same planned Exp463 record without result materialization or promotion.",
            symbol="\"Exp463\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "champion_ledger",
            "CHAMPION_LEDGER.json",
            "Caller: champion review. Effect: appends planned Exp463 history while preserving champion, pending candidate, routing, and promotion.",
            symbol="\"experiment\": \"Exp463\"",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "governance_regression",
            governance_test,
            "Caller: Task 1 regression. Effect: rejects missing governance surfaces, ledger drift, nonzero official lifecycle, label access, root presence, map mismatch, and champion/pending mutation.",
            symbol="test_task1_governance_antigravity",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "map_builder",
            "scratch/build_codemap_exp463_antigravity.py",
            "Caller: governance validation. Effect: builds and validates the bounded graph, resolving evidence, exact HTML-embedded JSON, and clean source-parent lock.",
            symbol="main_antigravity_validate_only_for_tests",
            tests=(governance_test,),
        ),
        _node_antigravity(
            "task2_unknown",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/config_antigravity.py",
            "unknown future Task 2 boundary; not implemented; callers/effects/tests are unknown until Task 2 review.",
            symbol="Task 2 — frozen Exp462 authority and TRAIN source",
            evidence_path=plan,
        ),
        _node_antigravity(
            "task3_unknown",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/scoring_antigravity.py",
            "unknown future Task 3 boundary; not implemented; callers/effects/tests are unknown until Task 3 review.",
            symbol="Task 3 — prefix-scaled ramp and seven method scores",
            evidence_path=plan,
        ),
        _node_antigravity(
            "task4_unknown",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/qualification_antigravity.py",
            "unknown future Task 4 boundary; not implemented; callers/effects/tests are unknown until Task 4 review.",
            symbol="Task 4 — blocked qualification, publication, semantic replay",
            evidence_path=plan,
        ),
        _node_antigravity(
            "task5_unknown",
            "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/cli_antigravity.py",
            "unknown future Task 5 boundary; not implemented; callers/effects/tests are unknown until Task 5 review.",
            symbol="Task 5 — closed CLI and two private TRAIN reproductions",
            evidence_path=plan,
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
            "evidence": {"path": "docs/EXP463_DASHBOARD_antigravity.html", "symbol": "M2 hydra_period_ramp_gated_max"},
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
                ("task2_unknown", "Task 2 — frozen Exp462 authority and TRAIN source"),
                ("task3_unknown", "Task 3 — prefix-scaled ramp and seven method scores"),
                ("task4_unknown", "Task 4 — blocked qualification, publication, semantic replay"),
                ("task5_unknown", "Task 5 — closed CLI and two private TRAIN reproductions"),
            )
        ],
    ]
    return {
        "edges": edges,
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "nodes": nodes,
        "schema": GRAPH_SCHEMA_ANTIGRAVITY,
        "source_parent_commit": SOURCE_PARENT_COMMIT_ANTIGRAVITY,
        "unknown_relationships": [],
    }


def _iter_tree_files_antigravity(relative_root: str) -> list[Path]:
    root = ROOT_ANTIGRAVITY / relative_root
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if path.name in {Path(item).name for item in TRIPLET_ANTIGRAVITY}:
            continue
        paths.append(path)
    return paths


def _fingerprint_tree_antigravity(relative_root: str) -> str:
    digest = hashlib.sha256()
    for path in _iter_tree_files_antigravity(relative_root):
        digest.update(path.relative_to(ROOT_ANTIGRAVITY).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _lock_antigravity(graph: dict[str, Any]) -> dict[str, Any]:
    tracked_paths = [
        "CHAMPION_LEDGER.json",
        "EXPERIMENT_CATALOG.json",
        "RESEARCH_STATE.json",
        "docs/EXP463_DASHBOARD_antigravity.html",
        "docs/EXP463_IMPLEMENTATION_REPORT_antigravity.html",
        "docs/EXP463_PREREGISTRATION_antigravity.html",
        "docs/EXP463_RUNBOOK_antigravity.html",
        "docs/superpowers/plans/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html",
        "docs/superpowers/specs/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity-design.html",
        "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity/tests/test_governance_antigravity.py",
        "scratch/build_codemap_exp463_antigravity.py",
    ]
    tracked = {
        path: _sha256_file_antigravity(ROOT_ANTIGRAVITY / path)
        for path in sorted(tracked_paths)
        if (ROOT_ANTIGRAVITY / path).is_file()
    }
    return {
        "generated_at": GENERATED_AT_ANTIGRAVITY,
        "module_fingerprints": {
            ".superpowers": _fingerprint_tree_antigravity(".superpowers"),
            "docs": _fingerprint_tree_antigravity("docs"),
            "experiments": _fingerprint_tree_antigravity("experiments"),
            "root": _fingerprint_tree_antigravity("."),
            "scratch": _fingerprint_tree_antigravity("scratch"),
        },
        "represented_paths": sorted(TRIPLET_ANTIGRAVITY),
        "schema": LOCK_SCHEMA_ANTIGRAVITY,
        "source_parent_commit": graph["source_parent_commit"],
        "tracked_fingerprints": tracked,
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
        "<title>Exp463 Task 1 코드맵</title>"
        "<style>body{max-width:1180px;margin:2rem auto;padding:0 1rem;font:15px/1.6 system-ui;color:#172033}"
        "code,pre{background:#eef2f7;padding:.12rem .3rem;overflow:auto}li{margin:.35rem 0}</style></head><body><main>"
        "<h1>Exp463 Task 1 코드맵</h1>"
        "<p>실제 Task 1 governance 노드와 resolving evidence를 기록한다. Task 2–5는 아직 구현되지 않아 unknown으로 표시한다.</p>"
        f"<h2>노드</h2><ul>{node_items}</ul>"
        "<h2>Machine-readable graph</h2>"
        f"<pre id=\"codemap-json-antigravity\" type=\"application/json\">{payload}</pre>"
        "</main></body></html>"
    )


def _validate_graph_antigravity(graph: dict[str, Any]) -> None:
    if graph.get("schema") != GRAPH_SCHEMA_ANTIGRAVITY:
        raise AssertionError("unexpected Exp463 graph schema")
    if graph.get("source_parent_commit") != SOURCE_PARENT_COMMIT_ANTIGRAVITY:
        raise AssertionError("unexpected source parent commit")
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
        "task2_unknown",
        "task3_unknown",
        "task4_unknown",
        "task5_unknown",
    }
    if not required.issubset(node_set):
        raise AssertionError(f"missing graph nodes: {sorted(required - node_set)}")
    for node in nodes:
        if not all(isinstance(node.get(key), str) and node.get(key) for key in ("id", "path", "role")):
            raise AssertionError("node identity/effect fields are required")
        if not isinstance(node.get("tests"), list):
            raise AssertionError("node tests must be a list")
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
        if not isinstance(evidence, dict) or not evidence.get("path") or not evidence.get("symbol"):
            raise AssertionError("edge evidence must resolve to path and symbol")
        if not (ROOT_ANTIGRAVITY / evidence["path"]).is_file():
            raise AssertionError(f"edge evidence path missing: {evidence['path']}")


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
