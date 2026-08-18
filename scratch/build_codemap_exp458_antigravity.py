#!/usr/bin/env python3
"""Build and validate the bounded Exp458 Task 2 rank-contract code map."""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import subprocess
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs" / "codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp458_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp458_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp458_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp458_antigravity.html",
    "docs/codemap/codemap_exp458_antigravity.json",
    "docs/codemap/codemap_exp458_antigravity.lock",
}
TRACKED_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "RESEARCH_STATE.json",
    "docs/EXP458_DASHBOARD_antigravity.html",
    "docs/EXP458_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP458_PREREGISTRATION_antigravity.html",
    "docs/superpowers/plans/2026-08-18-exp458-ecg-normal2-rank-confidence-routing-antigravity.html",
    "docs/superpowers/specs/2026-08-18-exp458-ecg-normal2-rank-confidence-routing-antigravity-design.html",
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/config_antigravity.py",
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/contracts_antigravity.py",
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/rank_auc_antigravity.py",
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/test_config_antigravity.py",
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/test_controlled_fork_antigravity.py",
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/test_governance_antigravity.py",
    "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/test_rank_auc_antigravity.py",
    "scratch/build_codemap_exp458_antigravity.py",
)
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"governs", "tests", "future_unknown"}


def canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def strict_json_antigravity(text: str) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates_antigravity)


def evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


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
        '<title>Exp458 Task 2 코드맵</title><style>body{max-width:1000px;margin:2rem auto;padding:0 1rem;font:16px/1.65 system-ui}code,pre{background:#eef2f7;padding:.15rem .3rem;overflow:auto}</style></head><body><main>'
        '<h1>Exp458 Task 2 rank-contract 코드맵</h1><p>Task 2까지 실제로 만든 고정 설정, rank contracts, AUC 계산기와 아직 만들지 않은 Task 3+ 경계를 분리한다. '
        'source, qualification, router, evaluation, CLI는 계속 future_unknown으로만 기록하며 구현 증거를 주장하지 않는다.</p>'
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

    plan = "docs/superpowers/plans/2026-08-18-exp458-ecg-normal2-rank-confidence-routing-antigravity.html"
    design = "docs/superpowers/specs/2026-08-18-exp458-ecg-normal2-rank-confidence-routing-antigravity-design.html"
    governance_test = (
        "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/"
        "test_governance_antigravity.py"
    )
    config_test = (
        "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/"
        "test_config_antigravity.py"
    )
    contracts_test = (
        "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/"
        "test_contracts_antigravity.py"
    )
    auc_test = (
        "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/"
        "test_rank_auc_antigravity.py"
    )
    controlled_fork_test = (
        "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity/tests/"
        "test_controlled_fork_antigravity.py"
    )
    builder = "scratch/build_codemap_exp458_antigravity.py"
    package = "experiments/exp458_ecg_normal2_rank_confidence_routing_antigravity"

    nodes: list[dict[str, object]] = [
        {
            "id": "approved_design",
            "path": design,
            "role": "Caller: Tasks 1–6 governance. Effect: freezes the normal_2-only AUC hypothesis, thresholds, and projected result boundary.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(design, "Exp458")],
        },
        {
            "id": "implementation_plan",
            "path": plan,
            "role": "Caller: Tasks 1–6. Effect: records that Tasks 1–2 are complete while Tasks 3–6 remain unchecked.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(plan, "Task 2: 고정 설정·계약·AUC 계산기")],
        },
        {
            "id": "preregistration",
            "path": "docs/EXP458_PREREGISTRATION_antigravity.html",
            "role": "Caller: governance review. Effect: records planned identity, AUC thresholds, projected result, and official-action firewall.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP458_PREREGISTRATION_antigravity.html", "planned_not_started")],
        },
        {
            "id": "implementation_report",
            "path": "docs/EXP458_IMPLEMENTATION_REPORT_antigravity.html",
            "role": "Caller: Task 2 closeout. Effect: explains the frozen contracts, AUC calculator, verification, and remaining Task 3+ boundary.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP458_IMPLEMENTATION_REPORT_antigravity.html", "Task 1과 Task 2를 완료했다")],
        },
        {
            "id": "dashboard",
            "path": "docs/EXP458_DASHBOARD_antigravity.html",
            "role": "Caller: reader-facing review. Effect: shows planned_not_started status, exact two-seed gate wording, Task 2 boundaries, and projected result without claiming an official outcome.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("docs/EXP458_DASHBOARD_antigravity.html", "Task 2/6")],
        },
        {
            "id": "research_state",
            "path": "RESEARCH_STATE.json",
            "role": "Caller: governance review. Effect: records Exp458 as planned_not_started with zero official activity and null hashes.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("RESEARCH_STATE.json", "\"exp458\"")],
        },
        {
            "id": "experiment_catalog",
            "path": "EXPERIMENT_CATALOG.json",
            "role": "Caller: catalog review. Effect: registers Exp458 documents and frozen scientific contract while keeping predecessor truth intact.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("EXPERIMENT_CATALOG.json", "\"id\": \"Exp458\"")],
        },
        {
            "id": "champion_ledger",
            "path": "CHAMPION_LEDGER.json",
            "role": "Caller: champion review. Effect: records Exp458 without changing champion or pending candidate.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp458\"")],
        },
        {
            "id": "governance_regression",
            "path": governance_test,
            "role": "Caller: Task 2 truth verification. Effect: rejects missing surfaces, ledger drift, fake official activity, or a dishonest Task 2 code map.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(governance_test, "test_exp458_is_planned_without_official_result_antigravity")],
        },
        {
            "id": "map_builder",
            "path": builder,
            "role": "Caller: Task 2 validation. Effect: validates HTML/JSON/lock parity, fingerprints, and Task 3+ future_unknown boundaries.",
            "tests": [governance_test],
            "evidence": [evidence_antigravity(builder, "validate_map_antigravity")],
        },
        {
            "id": "frozen_config",
            "path": f"{package}/config_antigravity.py",
            "role": "Caller: contracts, qualification, governance review. Effect: freezes Exp458 identity, exact dataset, seed order, thresholds, and structure gate.",
            "tests": [config_test, controlled_fork_test],
            "evidence": [evidence_antigravity(f"{package}/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")],
        },
        {
            "id": "result_contracts",
            "path": f"{package}/contracts_antigravity.py",
            "role": "Caller: future Tasks 3–4 qualification flow. Effect: validates exact normal_2 rank evidence, pooled AUC semantics, and both-seed qualification truth.",
            "tests": [contracts_test],
            "evidence": [evidence_antigravity(f"{package}/contracts_antigravity.py", "RankSeedEvidenceAntigravity")],
        },
        {
            "id": "rank_auc",
            "path": f"{package}/rank_auc_antigravity.py",
            "role": "Caller: future Task 3 qualification flow. Effect: computes Mann-Whitney style AUC with the frozen tie rule and exact score direction.",
            "tests": [auc_test],
            "evidence": [evidence_antigravity(f"{package}/rank_auc_antigravity.py", "mann_whitney_auc_antigravity")],
        },
        {
            "id": "train_source",
            "path": f"{package}/sources_antigravity.py",
            "role": "Caller: future Task 3. Effect: will load the exact normal_2 TRAIN source without labels.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
        {
            "id": "train_qualification",
            "path": f"{package}/qualification_antigravity.py",
            "role": "Caller: future Tasks 3–4. Effect: will compute the two-seed AUC qualification for normal_2 only.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
        {
            "id": "router",
            "path": f"{package}/router_antigravity.py",
            "role": "Caller: future Tasks 4–6. Effect: will combine AUC and structure gates into the normal_2 route decision.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
        {
            "id": "evaluation",
            "path": f"{package}/evaluation_antigravity.py",
            "role": "Caller: future Tasks 4–6. Effect: will recompute the frozen four comparisons and final criteria.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
        {
            "id": "artifacts",
            "path": f"{package}/artifacts_antigravity.py",
            "role": "Caller: future Tasks 4–6. Effect: will publish the canonical 9-leaf result inventory.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
        {
            "id": "replay",
            "path": f"{package}/replay_antigravity.py",
            "role": "Caller: future Tasks 4–6. Effect: will replay saved AUC, route, and comparison evidence from source data.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
        {
            "id": "dashboard_export",
            "path": f"{package}/dashboard_export_antigravity.py",
            "role": "Caller: future Tasks 4–6. Effect: will expose the reviewed dashboard payload only after semantic validation.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
        {
            "id": "cli",
            "path": f"{package}/cli_antigravity.py",
            "role": "Caller: future Tasks 5–6 operator. Effect: will expose the closed official commands after review.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
        {
            "id": "cli_launcher",
            "path": "scratch/run_exp458_ecg_normal2_rank_confidence_routing_antigravity.py",
            "role": "Caller: future Task 5 outside-CWD operator. Effect: will resolve the repository before importing only the package CLI.",
            "tests": [],
            "evidence": [],
            "status": "future_unknown",
        },
    ]

    edges = [
        {
            "from": "approved_design",
            "to": "preregistration",
            "type": "governs",
            "evidence": evidence_antigravity(
                "docs/EXP458_PREREGISTRATION_antigravity.html",
                "planned_not_started",
            ),
        },
        {
            "from": "implementation_plan",
            "to": "dashboard",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "Step 3: 최소 HTML·ledger·map 작성"),
        },
        {
            "from": "governance_regression",
            "to": "research_state",
            "type": "tests",
            "evidence": evidence_antigravity(
                governance_test,
                "test_exp458_is_planned_without_official_result_antigravity",
            ),
        },
        {
            "from": "governance_regression",
            "to": "dashboard",
            "type": "tests",
            "evidence": evidence_antigravity(
                governance_test,
                "test_dashboard_plan_and_map_stay_truthful_antigravity",
            ),
        },
        {
            "from": "approved_design",
            "to": "frozen_config",
            "type": "governs",
            "evidence": evidence_antigravity(design, "예정 패키지"),
        },
        {
            "from": "implementation_plan",
            "to": "result_contracts",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "Task 2: 고정 설정·계약·AUC 계산기"),
        },
        {
            "from": "implementation_plan",
            "to": "rank_auc",
            "type": "governs",
            "evidence": evidence_antigravity(plan, "Task 2: 고정 설정·계약·AUC 계산기"),
        },
        {
            "from": "frozen_config",
            "to": "result_contracts",
            "type": "governs",
            "evidence": evidence_antigravity(
                f"{package}/contracts_antigravity.py",
                "FROZEN_CONFIG_ANTIGRAVITY",
            ),
        },
        {
            "from": "rank_auc",
            "to": "result_contracts",
            "type": "governs",
            "evidence": evidence_antigravity(
                "docs/EXP458_IMPLEMENTATION_REPORT_antigravity.html",
                "(anomaly&gt;normal wins + 0.5*ties)/(n_normal*n_anomaly)",
            ),
        },
        {
            "from": "implementation_plan",
            "to": "train_source",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 3: TRAIN 점수와 normal_2 자격 평가"),
        },
        {
            "from": "implementation_plan",
            "to": "train_qualification",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 3: TRAIN 점수와 normal_2 자격 평가"),
        },
        {
            "from": "implementation_plan",
            "to": "router",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 4: 라우터·평가·9-leaf 결과·재생·대시보드"),
        },
        {
            "from": "implementation_plan",
            "to": "evaluation",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 4: 라우터·평가·9-leaf 결과·재생·대시보드"),
        },
        {
            "from": "implementation_plan",
            "to": "artifacts",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 4: 라우터·평가·9-leaf 결과·재생·대시보드"),
        },
        {
            "from": "implementation_plan",
            "to": "replay",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 4: 라우터·평가·9-leaf 결과·재생·대시보드"),
        },
        {
            "from": "implementation_plan",
            "to": "dashboard_export",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 4: 라우터·평가·9-leaf 결과·재생·대시보드"),
        },
        {
            "from": "implementation_plan",
            "to": "cli",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 5: 닫힌 CLI·후보 재현성·최종 검토"),
        },
        {
            "from": "implementation_plan",
            "to": "cli_launcher",
            "type": "future_unknown",
            "evidence": evidence_antigravity(plan, "Task 5: 닫힌 CLI·후보 재현성·최종 검토"),
        },
    ]
    unknown_relationships = [
        {"from": "implementation_plan", "to": "train_source", "reason": "Task 3 source loader is planned but not implemented yet."},
        {"from": "implementation_plan", "to": "train_qualification", "reason": "Task 3 qualification flow is planned but not implemented yet."},
        {"from": "implementation_plan", "to": "router", "reason": "Task 4 router is planned but not implemented yet."},
        {"from": "implementation_plan", "to": "evaluation", "reason": "Task 4 evaluator is planned but not implemented yet."},
        {"from": "implementation_plan", "to": "artifacts", "reason": "Task 4 publication layer is planned but not implemented yet."},
        {"from": "implementation_plan", "to": "replay", "reason": "Task 4 replay path is planned but not implemented yet."},
        {"from": "implementation_plan", "to": "dashboard_export", "reason": "Task 4 dashboard exporter is planned but not implemented yet."},
        {"from": "implementation_plan", "to": "cli", "reason": "Task 5 CLI is planned but not implemented yet."},
        {"from": "implementation_plan", "to": "cli_launcher", "reason": "Task 5 launcher is planned but not implemented yet."},
    ]
    graph = {
        "schema": "exp458_task2_rank_contracts_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": unknown_relationships,
    }
    lock = {
        "schema": "exp458_task2_rank_contracts_codemap_lock_v1",
        "source_parent_commit": source_commit,
        "working_tree_dirty": False,
        "represented_paths": sorted(TRIPLET_ANTIGRAVITY),
        "generated_at": generated_at,
        "module_fingerprints": module_fingerprints_antigravity(),
        "tracked_fingerprints": tracked_fingerprints_antigravity(),
    }
    return graph, lock, html_antigravity(graph)


def validate_evidence_antigravity(item: object) -> None:
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("symbol"), str)
    ):
        raise ValueError("invalid evidence")
    path = ROOT_ANTIGRAVITY / item["path"]
    if not path.is_file() or item["symbol"] not in path.read_text(encoding="utf-8"):
        raise ValueError("evidence not found")


def validate_map_antigravity(
    graph: object,
    lock: object,
    rendered_html: str,
) -> None:
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph and lock must be objects")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    unknowns = graph.get("unknown_relationships")
    if (
        not isinstance(nodes, list)
        or not isinstance(edges, list)
        or not isinstance(unknowns, list)
        or len(nodes) > 24
    ):
        raise ValueError("invalid bounded topology")

    ids: set[str] = set()
    future_ids: set[str] = set()
    for node in nodes:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("id"), str)
            or node["id"] in ids
        ):
            raise ValueError("invalid node")
        ids.add(node["id"])
        if node.get("status") == "future_unknown":
            future_ids.add(node["id"])
            if node.get("evidence") != [] or node.get("tests") != []:
                raise ValueError("future node claims evidence or tests")
            if not isinstance(node.get("path"), str) or (ROOT_ANTIGRAVITY / node["path"]).exists():
                raise ValueError("future node path already exists")
            continue
        if (
            not isinstance(node.get("path"), str)
            or not (ROOT_ANTIGRAVITY / node["path"]).is_file()
            or not isinstance(node.get("evidence"), list)
            or not node["evidence"]
            or not isinstance(node.get("tests"), list)
            or not node["tests"]
        ):
            raise ValueError("missing known evidence or tests")
        for item in node["evidence"]:
            validate_evidence_antigravity(item)

    unknown_pairs = {
        (item.get("from"), item.get("to"))
        for item in unknowns
        if isinstance(item, dict) and isinstance(item.get("reason"), str)
    }
    future_pairs: set[tuple[object, object]] = set()
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or edge.get("from") not in ids
            or edge.get("to") not in ids
            or edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY
        ):
            raise ValueError("invalid edge")
        validate_evidence_antigravity(edge.get("evidence"))
        pair = (edge["from"], edge["to"])
        if edge["from"] in future_ids or edge["to"] in future_ids:
            if edge["type"] != "future_unknown" or pair not in unknown_pairs:
                raise ValueError("unmarked future relationship")
            future_pairs.add(pair)
        elif edge["type"] == "future_unknown":
            raise ValueError("known edge marked future")
    if unknown_pairs != future_pairs:
        raise ValueError("unknown relationship mismatch")

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
            strict_json_antigravity(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8")),
            strict_json_antigravity(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8")),
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
