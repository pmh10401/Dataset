#!/usr/bin/env python3
"""Build the bounded, same-parent Exp451 governance code map."""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
MAP_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs" / "codemap"
JSON_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp451_antigravity.json"
LOCK_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp451_antigravity.lock"
HTML_PATH_ANTIGRAVITY = MAP_DIR_ANTIGRAVITY / "codemap_exp451_antigravity.html"
TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp451_antigravity.html",
    "docs/codemap/codemap_exp451_antigravity.json",
    "docs/codemap/codemap_exp451_antigravity.lock",
}
ALLOWED_EDGE_TYPES_ANTIGRAVITY = {"reads", "governs", "tests", "future_unknown"}


def _canonical_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _git_output_antigravity(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_ANTIGRAVITY, text=True).strip()


def _tracked_paths_antigravity() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT_ANTIGRAVITY)
    return sorted(
        Path(entry.decode("utf-8"))
        for entry in raw.split(b"\0")
        if entry and entry.decode("utf-8") not in TRIPLET_ANTIGRAVITY
    )


def _tracked_fingerprints_antigravity() -> dict[str, str]:
    return {
        relative.as_posix(): hashlib.sha256((ROOT_ANTIGRAVITY / relative).read_bytes()).hexdigest()
        for relative in _tracked_paths_antigravity()
    }


def _module_fingerprints_antigravity() -> dict[str, str]:
    grouped: dict[str, list[Path]] = {"root": []}
    for relative in _tracked_paths_antigravity():
        key = "root" if len(relative.parts) == 1 else relative.parts[0]
        grouped.setdefault(key, []).append(relative)
    result: dict[str, str] = {}
    for key, paths in sorted(grouped.items()):
        digest = hashlib.sha256()
        for relative in sorted(paths):
            contents = (ROOT_ANTIGRAVITY / relative).read_bytes()
            digest.update(relative.as_posix().encode("utf-8") + b"\0")
            digest.update(str(len(contents)).encode("ascii") + b"\0" + contents + b"\0")
        result[key] = digest.hexdigest()
    return result


def _evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def _is_allowed_source_parent_antigravity(source_parent: str) -> bool:
    """Allow a clean functional HEAD or exactly its map-only child."""
    head = _git_output_antigravity("rev-parse", "HEAD")
    if source_parent == head:
        return True
    parent = _git_output_antigravity("rev-parse", "HEAD^")
    if source_parent != parent:
        return False
    changed = set(_git_output_antigravity("diff", "--name-only", "HEAD^", "HEAD").splitlines())
    return changed == TRIPLET_ANTIGRAVITY


def _self_contained_html_antigravity(graph: dict[str, object]) -> str:
    graph_json = _canonical_bytes_antigravity(graph).decode("utf-8")
    rendered = html_module.escape(graph_json)
    return f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Exp451 코드맵</title><style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:2rem auto;padding:0 1rem}}code,pre{{background:#f1f4f8;padding:.15rem .3rem;border-radius:4px;overflow:auto}}li{{margin:.35rem 0}}</style></head>
<body><h1>Exp451 거버넌스 코드맵</h1><p>Exp449는 active로 보존되고, Exp451은 planned sibling이다. Task 3은 official CPU RDST와 TRAIN-only LocalGap 점수 및 fail-closed backend gate를 작은 fixture source와 focused test 증거로 연결한다. official lifecycle·labels 경로는 여전히 없다.</p>
<h2>노드</h2><ul>{''.join(f'<li><code>{html_module.escape(str(node["path"]))}</code> — {html_module.escape(str(node["role"]))}</li>' for node in graph["nodes"])}</ul>
<h2>관계</h2><pre id=\"codemap-json-antigravity\" type=\"application/json\">{rendered}</pre></body></html>"""


def build_map_antigravity(source_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    """Return the Exp451 graph, parent-bound lock, and self-contained HTML."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    if subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT_ANTIGRAVITY, check=False).returncode:
        raise ValueError("source_commit does not resolve to a commit")
    if not generated_at.endswith("Z"):
        raise ValueError("generated_at must be UTC ISO-8601 ending in Z")

    governance_test = "experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_governance_antigravity.py"
    nodes: list[dict[str, object]] = [
        {"id": "approved_design", "path": "docs/superpowers/specs/2026-08-13-exp451-rdst-local-gap-qualification-antigravity-design.html", "role": "승인된 RDST + LocalGap 수치와 post-hoc 경계의 단일 설계 근거.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/superpowers/specs/2026-08-13-exp451-rdst-local-gap-qualification-antigravity-design.html", "Exp451: RDST + LocalGap 자격검증 설계")]},
        {"id": "implementation_plan", "path": "docs/superpowers/plans/2026-08-13-exp451-rdst-local-gap-qualification-antigravity.html", "role": "Task 1의 RED/GREEN, map-only child 및 source-parent 규약.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/superpowers/plans/2026-08-13-exp451-rdst-local-gap-qualification-antigravity.html", "Task 1: 거버넌스 동결과 Exp451 코드맵")]},
        {"id": "preregistration", "path": "docs/EXP451_PREREGISTRATION_antigravity_KO.html", "role": "동결 과학 설정, label firewall, backend gate, no-promotion 계약.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/EXP451_PREREGISTRATION_antigravity_KO.html", "post-hoc method qualification")]},
        {"id": "runbook", "path": "docs/EXP451_RUNBOOK_antigravity_KO.html", "role": "별도 승인 전 official lifecycle 금지와 미래 단일 label session 경계.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/EXP451_RUNBOOK_antigravity_KO.html", "현재 금지")]},
        {"id": "implementation_report", "path": "docs/EXP451_IMPLEMENTATION_REPORT_antigravity.html", "role": "Task 1 구현 범위와 no-official-run 상태 기록.", "tests": [governance_test], "evidence": [_evidence_antigravity("docs/EXP451_IMPLEMENTATION_REPORT_antigravity.html", "Task 1은 거버넌스")]} ,
        {"id": "research_state", "path": "RESEARCH_STATE.json", "role": "Exp449 active terminal truth를 보존하고 Exp451을 Exp448-only scientific-input planned sibling으로 등록.", "tests": [governance_test], "evidence": [_evidence_antigravity("RESEARCH_STATE.json", "\"planned_not_started\""), _evidence_antigravity("RESEARCH_STATE.json", "\"official_readiness_passed_evaluation_failed_immutable_no_retry\"")]},
        {"id": "experiment_catalog", "path": "EXPERIMENT_CATALOG.json", "role": "Exp451 identity, roots, frozen RDST/LocalGap contract catalog.", "tests": [governance_test], "evidence": [_evidence_antigravity("EXPERIMENT_CATALOG.json", "\"rdst_local_gap_qualification\"")]},
        {"id": "champion_ledger", "path": "CHAMPION_LEDGER.json", "role": "champion/pending candidate 불변성과 Exp451 no-promotion 기록.", "tests": [governance_test], "evidence": [_evidence_antigravity("CHAMPION_LEDGER.json", "\"experiment\": \"Exp451\"")]},
        {"id": "governance_regression", "path": governance_test, "role": "Exp449 protected-state SHA와 Exp451 planned/label-free 상태를 독립 검증.", "tests": [governance_test], "evidence": [_evidence_antigravity(governance_test, "protected_exp449_state_sha256_antigravity")]},
        {"id": "map_builder", "path": "scratch/build_codemap_exp451_antigravity.py", "role": "30-node 상한, evidence/ref/unknown/fingerprint 검증과 same-parent lock 생성.", "tests": [governance_test], "evidence": [_evidence_antigravity("scratch/build_codemap_exp451_antigravity.py", "def validate_map_antigravity")]},
        {"id": "exp449_predecessor_boundary", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "role": "향후 재사용 가능한 것은 Exp449 LocalGap code semantics뿐이며, failed evaluation artifact는 scientific input이 아니다.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_scoring_antigravity.py"], "evidence": [_evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "def score_candidate_antigravity")]},
        {"id": "exp449_terminal_outcome", "path": "RESEARCH_STATE.json", "role": "Exp449의 one-session label evaluation failure와 permanent stop은 governance-only evidence이며 Exp451 input으로 재사용하지 않는다.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_readiness_antigravity.py"], "evidence": [_evidence_antigravity("RESEARCH_STATE.json", "frozen prediction indices differ from evidence"), _evidence_antigravity("RESEARCH_STATE.json", "\"permanent_stop\": true")]},
        {"id": "rdst_config", "path": "experiments/exp451_rdst_local_gap_qualification_antigravity/config_antigravity.py", "role": "Exp451 identity, root, RDST·LocalGap 수치와 CPU/thread 규칙을 immutable exact-key authority로 동결.", "tests": ["experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_config_antigravity.py"], "evidence": [_evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY"), _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/config_antigravity.py", "decode_frozen_config_antigravity")]},
        {"id": "rdst_contracts", "path": "experiments/exp451_rdst_local_gap_qualification_antigravity/contracts_antigravity.py", "role": "label 없는 dataset bundle, score snapshot, canonical frozen prediction authority를 exact type·immutable bytes로 강제.", "tests": ["experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_contracts_antigravity.py"], "evidence": [_evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/contracts_antigravity.py", "DatasetBundleAntigravity"), _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/contracts_antigravity.py", "FrozenPredictionAuthorityAntigravity")]},
        {"id": "rdst_method_audit", "path": "experiments/exp451_rdst_local_gap_qualification_antigravity/method_audit_antigravity.py", "role": "고정 interpreter subprocess로 aeon class/signature/source SHA와 y=None deterministic fixture를 fail-closed audit.", "tests": ["experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_method_audit_antigravity.py"], "evidence": [_evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/method_audit_antigravity.py", "audit_rdst_antigravity"), _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/method_audit_antigravity.py", "RdstAuditResultAntigravity")]},
        {"id": "official_rdst_representation", "path": "experiments/exp451_rdst_local_gap_qualification_antigravity/rdst_antigravity.py", "role": "공식 aeon RDST를 CPU n_jobs=1, y=None, (rows,1,length) 경계로만 감싼 fresh representation.", "tests": ["experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_rdst_antigravity.py", "experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_scoring_antigravity.py"], "evidence": [_evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/rdst_antigravity.py", "RDSTRepresentationAntigravity"), _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/rdst_antigravity.py", "make_rdst_antigravity")]},
        {"id": "localgap_scoring", "path": "experiments/exp451_rdst_local_gap_qualification_antigravity/scoring_antigravity.py", "role": "정상 TRAIN-only 5-fold OOF와 fresh full RDST/LocalGap 점수, 두 seed median을 label 없이 계산.", "tests": ["experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_scoring_antigravity.py"], "evidence": [_evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/scoring_antigravity.py", "score_candidate_antigravity"), _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/scoring_antigravity.py", "aggregate_seed_scores_antigravity")]},
        {"id": "localgap_backend_gate", "path": "experiments/exp451_rdst_local_gap_qualification_antigravity/backend_gate_antigravity.py", "role": "MLX/MPS optional runtime probe를 lazy하게 기록하고 parity·error·inclusive speedup gate 미충족 시 CPU로 fail-close.", "tests": ["experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_backend_gate_antigravity.py"], "evidence": [_evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/backend_gate_antigravity.py", "decide_backend_antigravity"), _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/backend_gate_antigravity.py", "build_local_gap_backend_antigravity")]},
    ]
    edges: list[dict[str, object]] = [
        {"from": "approved_design", "to": "preregistration", "type": "governs", "evidence": _evidence_antigravity("docs/EXP451_PREREGISTRATION_antigravity_KO.html", "설계 SHA256")},
        {"from": "implementation_plan", "to": "map_builder", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/plans/2026-08-13-exp451-rdst-local-gap-qualification-antigravity.html", "build_map_antigravity(source_commit")},
        {"from": "research_state", "to": "preregistration", "type": "reads", "evidence": _evidence_antigravity("RESEARCH_STATE.json", "EXP451_PREREGISTRATION_antigravity_KO.html")},
        {"from": "governance_regression", "to": "research_state", "type": "tests", "evidence": _evidence_antigravity(governance_test, "RESEARCH_STATE.json")},
        {"from": "governance_regression", "to": "preregistration", "type": "tests", "evidence": _evidence_antigravity(governance_test, "EXP451_PREREGISTRATION_antigravity_KO.html")},
        {"from": "experiment_catalog", "to": "research_state", "type": "governs", "evidence": _evidence_antigravity("EXPERIMENT_CATALOG.json", "Exp451")},
        {"from": "champion_ledger", "to": "research_state", "type": "reads", "evidence": _evidence_antigravity("CHAMPION_LEDGER.json", "Exp451")},
        {"from": "exp449_terminal_outcome", "to": "research_state", "type": "reads", "evidence": _evidence_antigravity("RESEARCH_STATE.json", "failed artifact is not reused")},
        {"from": "approved_design", "to": "rdst_config", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/specs/2026-08-13-exp451-rdst-local-gap-qualification-antigravity-design.html", "동결할 과학 설정")},
        {"from": "preregistration", "to": "rdst_config", "type": "governs", "evidence": _evidence_antigravity("docs/EXP451_PREREGISTRATION_antigravity_KO.html", "max_shapelets=1024")},
        {"from": "rdst_config", "to": "rdst_method_audit", "type": "reads", "evidence": _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/method_audit_antigravity.py", "FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY")},
        {"from": "approved_design", "to": "rdst_contracts", "type": "governs", "evidence": _evidence_antigravity("docs/superpowers/specs/2026-08-13-exp451-rdst-local-gap-qualification-antigravity-design.html", "prediction artifact에")},
        {"from": "rdst_config", "to": "official_rdst_representation", "type": "governs", "evidence": _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/rdst_antigravity.py", "max_shapelets=1024")},
        {"from": "rdst_method_audit", "to": "official_rdst_representation", "type": "governs", "evidence": _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/rdst_antigravity.py", "RandomDilatedShapeletTransform")},
        {"from": "official_rdst_representation", "to": "localgap_scoring", "type": "reads", "evidence": _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/scoring_antigravity.py", "make_rdst_antigravity")},
        {"from": "localgap_backend_gate", "to": "localgap_scoring", "type": "reads", "evidence": _evidence_antigravity("experiments/exp451_rdst_local_gap_qualification_antigravity/scoring_antigravity.py", "build_local_gap_backend_antigravity")},
    ]
    unknown_relationships: list[dict[str, str]] = []
    graph: dict[str, object] = {
        "schema": "exp451_codemap_v1",
        "generated_at": generated_at,
        "source_parent_commit": source_commit,
        "topology_note": "이 bounded triplet은 Exp449 active terminal truth와 Exp448 stable-26 scientific-input 경계를 보존한다. Task 2는 immutable config, label-free contracts, fixed-interpreter RDST audit와 focused tests를 source evidence로 연결하며, 아직 실행·점수화·label 경로를 주장하지 않는다.",
        "nodes": nodes,
        "edges": edges,
        "unknown_relationships": unknown_relationships,
    }
    lock: dict[str, object] = {
        "schema": "exp451_codemap_same_commit_v1",
        "source_parent_commit": source_commit,
        "working_tree_dirty": bool(_git_output_antigravity("status", "--porcelain")),
        "represented_paths": sorted(TRIPLET_ANTIGRAVITY),
        "generated_at": generated_at,
        "fingerprint_algorithm": "sha256-v1: tracked bytes excluding exactly the Exp451 map triplet",
        "module_fingerprints": _module_fingerprints_antigravity(),
        "tracked_fingerprints": _tracked_fingerprints_antigravity(),
    }
    return graph, lock, _self_contained_html_antigravity(graph)


def validate_map_antigravity(graph: object, lock: object, html: str) -> None:
    """Reject incomplete evidence, invalid links, drift, or unmarked future claims."""
    if not isinstance(graph, dict) or not isinstance(lock, dict) or not isinstance(html, str):
        raise ValueError("graph, lock, and html must have canonical types")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    unknowns = graph.get("unknown_relationships")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(unknowns, list):
        raise ValueError("graph topology is missing")
    if len(nodes) > 30:
        raise ValueError("more than 30 primary nodes")
    ids: set[str] = set()
    future_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not isinstance(node.get("path"), str):
            raise ValueError("invalid node")
        node_id = node["id"]
        if node_id in ids:
            raise ValueError("duplicate node id")
        ids.add(node_id)
        if node.get("status") == "future_unknown":
            future_ids.add(node_id)
            if node.get("evidence") not in ([], None):
                raise ValueError("future unknown node cannot claim evidence")
            continue
        path = ROOT_ANTIGRAVITY / node["path"]
        if not path.is_file():
            raise ValueError(f"invalid node path: {node['path']}")
        evidence = node.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("missing node evidence")
        for item in evidence:
            _validate_evidence_antigravity(item)
    unknown_pairs: set[tuple[str, str]] = set()
    for unknown in unknowns:
        if not isinstance(unknown, dict) or not isinstance(unknown.get("from"), str) or not isinstance(unknown.get("to"), str) or not isinstance(unknown.get("reason"), str):
            raise ValueError("unmarked unknown relationship")
        if unknown["from"] not in ids or unknown["to"] not in ids:
            raise ValueError("invalid unknown reference")
        unknown_pairs.add((unknown["from"], unknown["to"]))
    future_edge_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids:
            raise ValueError("invalid edge reference")
        if edge.get("type") not in ALLOWED_EDGE_TYPES_ANTIGRAVITY:
            raise ValueError("invalid edge type")
        _validate_evidence_antigravity(edge.get("evidence"))
        pair = (edge["from"], edge["to"])
        touches_future = edge["from"] in future_ids or edge["to"] in future_ids
        if touches_future:
            if edge["type"] != "future_unknown" or pair not in unknown_pairs:
                raise ValueError("unmarked unknown relationship")
            future_edge_pairs.add(pair)
        elif edge["type"] == "future_unknown" or pair in unknown_pairs:
            raise ValueError("unknown relationship asserted as known")
    if unknown_pairs != future_edge_pairs:
        raise ValueError("unmarked unknown relationship")
    if not re.fullmatch(r"[0-9a-f]{40}", str(lock.get("source_parent_commit", ""))):
        raise ValueError("invalid source parent")
    if lock.get("source_parent_commit") != graph.get("source_parent_commit"):
        raise ValueError("graph/lock source parent mismatch")
    if not _is_allowed_source_parent_antigravity(str(lock["source_parent_commit"])):
        raise ValueError("invalid source parent relationship")
    if lock.get("working_tree_dirty") is not False:
        raise ValueError("same-parent lock must be generated from a clean functional commit")
    if lock.get("represented_paths") != sorted(TRIPLET_ANTIGRAVITY):
        raise ValueError("invalid represented paths")
    if lock.get("tracked_fingerprints") != _tracked_fingerprints_antigravity() or lock.get("module_fingerprints") != _module_fingerprints_antigravity():
        raise ValueError("fingerprint drift")
    match = re.search(r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>', html, flags=re.DOTALL)
    if not match:
        raise ValueError("HTML lacks embedded graph")
    embedded = json.loads(html_module.unescape(match.group(1)))
    if _canonical_bytes_antigravity(embedded) != _canonical_bytes_antigravity(graph):
        raise ValueError("HTML/JSON mismatch")


def _validate_evidence_antigravity(item: object) -> None:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("symbol"), str) or not item["symbol"]:
        raise ValueError("missing evidence")
    path = ROOT_ANTIGRAVITY / item["path"]
    if not path.is_file() or item["symbol"] not in path.read_text(encoding="utf-8"):
        raise ValueError("invalid evidence reference")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at", default="2026-08-13T00:00:00Z")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        graph = json.loads(JSON_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
        lock = json.loads(LOCK_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
        validate_map_antigravity(graph, lock, HTML_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
        return
    if not args.source_commit:
        parser.error("--source-commit is required unless --validate-only is used")
    if _git_output_antigravity("status", "--porcelain"):
        raise ValueError("map generation requires a clean functional commit")
    graph, lock, rendered_html = build_map_antigravity(args.source_commit, args.generated_at)
    validate_map_antigravity(graph, lock, rendered_html)
    MAP_DIR_ANTIGRAVITY.mkdir(parents=True, exist_ok=True)
    JSON_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(graph) + b"\n")
    LOCK_PATH_ANTIGRAVITY.write_bytes(_canonical_bytes_antigravity(lock) + b"\n")
    HTML_PATH_ANTIGRAVITY.write_text(rendered_html, encoding="utf-8")


if __name__ == "__main__":
    main_antigravity()
