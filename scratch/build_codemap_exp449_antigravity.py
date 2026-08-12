#!/usr/bin/env python3
"""Build and validate the Exp449 map after verified Task 3 closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "docs/codemap"
JSON_PATH = MAP_DIR / "codemap_exp449_antigravity.json"
LOCK_PATH = MAP_DIR / "codemap_exp449_antigravity.lock"
HTML_PATH = MAP_DIR / "codemap_exp449_antigravity.html"
SCANNED_TOP_LEVEL = (".superpowers", "docs", "experiments", "operational", "reports", "scratch", "scripts", "vercel-dashboard")
EXCLUDED_PARTS = {".git", ".worktrees", "vendor", "node_modules", "build", "cache", "dist", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "outputs", "results"}
EDGE_TYPES = {"imports", "reads", "governs", "future_unknown"}
_MAP_ONLY_PATHS_ANTIGRAVITY = {"scratch/build_codemap_exp449_antigravity.py"}


def strict_json_antigravity(text: str) -> Any:
    def reject_duplicate_antigravity(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicate_antigravity)


def tracked_paths_antigravity() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths: list[Path] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        parts = relative.parts
        if any(part in EXCLUDED_PARTS for part in parts) or parts[:2] == ("docs", "codemap"):
            continue
        if len(parts) == 1 or parts[0] in SCANNED_TOP_LEVEL:
            paths.append(relative)
    return sorted(paths)


def fingerprint_modules_antigravity() -> dict[str, str]:
    grouped: dict[str, list[Path]] = {"root": []}
    for name in SCANNED_TOP_LEVEL:
        grouped[name] = []
    for relative in tracked_paths_antigravity():
        key = "root" if len(relative.parts) == 1 else relative.parts[0]
        if key in grouped:
            grouped[key].append(relative)
    result: dict[str, str] = {}
    for key, paths in grouped.items():
        digest = hashlib.sha256()
        for relative in sorted(paths):
            contents = (ROOT / relative).read_bytes()
            encoded_path = relative.as_posix().encode("utf-8")
            digest.update(encoded_path + b"\0" + str(len(contents)).encode("ascii") + b"\0" + contents + b"\0")
        result[key] = digest.hexdigest()
    return result


def evidence_antigravity(path: str, symbol: str) -> dict[str, str]:
    return {"path": path, "symbol": symbol}


def _git_commit_exists_antigravity(commit: str) -> bool:
    """Return whether a full source binding resolves to a commit object in this repository."""
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return False
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0


def _is_map_only_commit_antigravity(commit: str) -> bool:
    """Recognize a map-only child without letting map contents fingerprint themselves."""
    paths = [path for path in subprocess.check_output(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit], cwd=ROOT, text=True).splitlines() if path]
    return bool(paths) and all(path.startswith("docs/codemap/") or path in _MAP_ONLY_PATHS_ANTIGRAVITY for path in paths)


def _expected_map_parent_antigravity() -> str | None:
    """Return the first non-map parent only when the checked-out HEAD is a map-only child."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not _is_map_only_commit_antigravity(head):
        return None
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT, text=True).strip()
    while _is_map_only_commit_antigravity(parent):
        parent = subprocess.check_output(["git", "rev-parse", f"{parent}^"], cwd=ROOT, text=True).strip()
    return parent


def build_map_antigravity(source_commit: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    """Return JSON graph, lock, and self-contained HTML for the same graph."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be lowercase 40-hex")
    if not _git_commit_exists_antigravity(source_commit):
        raise ValueError("source commit does not exist")
    nodes: list[dict[str, object]] = [
        {"id": "research_control", "path": "RESEARCH_STATE.json", "role": "Planned Exp449 active state with zero label access and closed official lifecycle.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("RESEARCH_STATE.json", '"exp449"')]},
        {"id": "champion_ledger", "path": "CHAMPION_LEDGER.json", "role": "Non-promotion ledger retaining the unchanged pending candidate.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("CHAMPION_LEDGER.json", '"experiment": "Exp449"')]},
        {"id": "exp449_preregistration", "path": "docs/EXP449_PREREGISTRATION_antigravity_KO.html", "role": "Frozen candidates, settings, success gates, label boundary, CPU policy, and no-promotion contract.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "official_label_sessions = 1")]},
        {"id": "exp449_runbook", "path": "docs/EXP449_RUNBOOK_antigravity_KO.html", "role": "Single-use future command order, seven thread variables, stop rules, and dashboard firewall.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_RUNBOOK_antigravity_KO.html", "AEON_NUM_THREADS=1")]},
        {"id": "project_execution_policy", "path": "AGENTS.md", "role": "Durable global rule: checked HTML-plan controls represent only completed implementation and review/verification boundaries.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("AGENTS.md", "In HTML implementation plans")]},
        {"id": "html_implementation_plan", "path": "docs/superpowers/plans/2026-08-12-exp449-residual-fn-method-qualification-antigravity.html", "role": "Self-contained Exp449 implementation plan; Task 5 implementation is mapped while its disabled plan controls remain unchanged pending final review.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-12-exp449-residual-fn-method-qualification-antigravity.html", "Task 4: Four Fixed Representations")]},
        {"id": "governance_regression", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "role": "Structural stdlib HTML-parser regression for Exp449 plan task/step completion state and global governance constraints.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "class PlanCheckboxParserAntigravity")]},
        {"id": "candidate_registry", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py", "role": "Frozen eight-family, two-candidate registry and pinned environment authority.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_config_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "immutable_contracts", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py", "role": "Frozen slots exact-type authorities for candidates, audits, bundles, baselines, and metrics.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_contracts_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py", "class MethodAuditAntigravity")]},
        {"id": "canonical_artifacts", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py", "role": "Canonical JSON/JSONL and create-only no-follow artifact inventory boundary used by immutable predecessor capture.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_artifacts_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py", "def artifact_map_bytes_antigravity")]},
        {"id": "method_audit", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "role": "Subprocess-only fixed-interpreter audit binding versions, source SHA256 values, and frozen public methods.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_method_audit_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "def audit_methods_antigravity")]},
        {"id": "exp448_authority", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "role": "Descriptor-pinned immutable Exp448 11-leaf authority, independently binding the source-free Exp446 stable-26 publication.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_predecessor_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "def capture_exp448_authority_antigravity")]},
        {"id": "label_free_source", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py", "role": "Read-only descriptor-pinned value loader and Task-6-only closure-private TEST-label session boundary. It consumes Task-5's exact authority validator before DB reachability.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_sources_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py", "def _frozen_prediction_order_antigravity")]},
        {"id": "representation_sources", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py", "role": "Four deterministic fixed aeon 1.4.0 CPU representation adapters, including Hydra-first scaled hybrid concatenation and no classifier head.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_representations_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py", "def make_representation_antigravity")]},
        {"id": "local_gap", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/local_gap_antigravity.py", "role": "Exp249-compatible robust-scaled LocalGap model under the Exp449 namespace.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_scoring_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/local_gap_antigravity.py", "class LocalGapModelAntigravity")]},
        {"id": "score_sources", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "role": "Sequential deterministic five-fold, reference-capped, chunked label-free scores; its frozen score contract is the future Task-5 prediction input.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_scoring_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "def score_candidate_antigravity")]},
        {"id": "acceleration_gate", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/mlx_gate_antigravity.py", "role": "Stage-specific immutable CPU/MLX/MPS parity and speed gate; absent MLX in the fixed environment retains CPU canonical scoring.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_mlx_gate_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/mlx_gate_antigravity.py", "def assess_mlx_antigravity")]},
        {"id": "frozen_prediction_lifecycle", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "role": "Task-5 fixed-root label-free 96-unit prediction authority: non-injectable 11-worker production route, parent-contiguous lifecycle-owned CommitSession commits, descriptor validator firewall, and one shared non-masking failure-audit boundary.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_prediction_antigravity.py", "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_progress_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "def _handle_lifecycle_failure_antigravity")]},
        {"id": "prediction_progress", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/progress_antigravity.py", "role": "Task-5 create-only candidate durability with a lifecycle-owned private CommitSession that retains only writer-returned provenance, derives split progress channels, and returns nonthrowing longest-prefix evidence.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_progress_antigravity.py", "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_prediction_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/progress_antigravity.py", "class PrefixValidationAntigravity")]},
        {"id": "local_status_projection", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/dashboard_export_antigravity.py", "role": "Task-5 local-only exact-twelve-field dashboard status boundary with recursive sensitive-value/type firewall and no scientific-module import.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_dashboard_export_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/dashboard_export_antigravity.py", "def public_status_antigravity")]},
    ]
    edges: list[dict[str, object]] = [
        {"from": "research_control", "to": "exp449_preregistration", "type": "reads", "evidence": evidence_antigravity("RESEARCH_STATE.json", "EXP449_PREREGISTRATION_antigravity_KO.html")},
        {"from": "research_control", "to": "exp449_runbook", "type": "reads", "evidence": evidence_antigravity("RESEARCH_STATE.json", "EXP449_RUNBOOK_antigravity_KO.html")},
        {"from": "champion_ledger", "to": "research_control", "type": "governs", "evidence": evidence_antigravity("CHAMPION_LEDGER.json", "pending_candidate_changed")},
        {"from": "project_execution_policy", "to": "html_implementation_plan", "type": "governs", "evidence": evidence_antigravity("AGENTS.md", "In HTML implementation plans")},
        {"from": "governance_regression", "to": "project_execution_policy", "type": "reads", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "Path(\"AGENTS.md\")")},
        {"from": "governance_regression", "to": "html_implementation_plan", "type": "reads", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "test_html_plan_records_only_verified_task_and_step_progress_antigravity")},
        {"from": "exp449_preregistration", "to": "candidate_registry", "type": "governs", "evidence": evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "global_fp_max = 314")},
        {"from": "method_audit", "to": "candidate_registry", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "config_antigravity import")},
        {"from": "method_audit", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "contracts_antigravity import")},
        {"from": "exp448_authority", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "contracts_antigravity import")},
        {"from": "exp448_authority", "to": "canonical_artifacts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "artifacts_antigravity import")},
        {"from": "exp448_authority", "to": "candidate_registry", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "config_antigravity import")},
        {"from": "label_free_source", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py", "contracts_antigravity import")},
        {"from": "candidate_registry", "to": "representation_sources", "type": "governs", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py", "minirocket_requested_kernels")},
        {"from": "score_sources", "to": "representation_sources", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "representations_antigravity import")},
        {"from": "score_sources", "to": "local_gap", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "local_gap_antigravity import")},
        {"from": "score_sources", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "contracts_antigravity import")},
        {"from": "score_sources", "to": "frozen_prediction_lifecycle", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "empirical_candidates")},
        {"from": "label_free_source", "to": "frozen_prediction_lifecycle", "type": "reads", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py", "validate_frozen_prediction_authority_antigravity")},
        {"from": "frozen_prediction_lifecycle", "to": "prediction_progress", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "progress_antigravity import")},
    ]
    graph: dict[str, object] = {"generated_at": generated_at, "generated_from_commit": source_commit, "scope": ["Exp449 verified Tasks 1–5: governance, immutable contracts, descriptor-pinned values-only source boundary, deterministic aeon score sources, CPU-canonical acceleration record, and frozen label-free prediction lifecycle."], "nodes": nodes, "edges": edges, "flows": [{"trigger": "Governance and fixed method inputs", "steps": ["exp449_preregistration", "candidate_registry", "method_audit", "immutable_contracts"], "outcome": "Frozen candidate and exact-type authority constrain Task-4 methods without label access, promotion, or routing."}, {"trigger": "Task-4 label-free score generation", "steps": ["label_free_source", "representation_sources", "local_gap", "score_sources", "acceleration_gate"], "outcome": "CPU aeon representations and Exp249 LocalGap return finite deterministic scores; MLX/MPS availability cannot approve acceleration without the full stage gate."}, {"trigger": "Task-5 durable prediction authority", "steps": ["exp448_authority", "label_free_source", "score_sources", "frozen_prediction_lifecycle", "prediction_progress"], "outcome": "A fixed 11-worker production route derives progress and failure exact-next evidence only from lifecycle-owned CommitSession state appended after durable writer success; one shared failure boundary independently records a nonthrowing descriptor prefix and audit outcome before Task 6 can open TEST labels, while status remains local-only."}], "unknown_relationships": ["Task 6 evaluation, replay, launcher, and external dashboard publication remain unimplemented."], "unknowns": ["The Task-6 label opener remains fail-closed until an official Task-5 prediction authority is present at its fixed root."]}
    lock: dict[str, object] = {"current_commit": source_commit, "working_tree_dirty": False, "generated_at": generated_at, "scanned_scope": ["tracked root files including AGENTS.md", *SCANNED_TOP_LEVEL], "excluded_directories": sorted(EXCLUDED_PARTS | {"docs/codemap"}), "fingerprint_algorithm": "sha256-v1: sort tracked repo-relative paths by module; append UTF-8 path, NUL, decimal byte length, NUL, current bytes, NUL; docs/codemap is excluded to prevent self-reference.", "module_fingerprints": fingerprint_modules_antigravity(), "changed_modules_since_prior_lock": ["docs", "experiments", "scratch"]}
    payload = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    html = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Exp449 Code Map</title><style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px/1.55 system-ui}pre{white-space:pre-wrap;overflow:auto;background:#eef2f7;padding:1rem}</style></head><body><main><h1>Exp449 Tasks 1–5 Code Map</h1><p>Task 5 frozen prediction lifecycle is implemented; Task 6 remains outside this map.</p><script id=\"codemap-data\" type=\"application/json\">" + payload + "</script><pre id=\"graph\"></pre></main><script>const data=JSON.parse(document.getElementById('codemap-data').textContent);document.getElementById('graph').textContent=JSON.stringify(data,null,2);</script></body></html>"
    return graph, lock, html


def validate_map_antigravity(graph: object, lock: object, html: str) -> None:
    """Reject missing evidence, unresolved refs, >20 primary nodes, and fingerprint drift."""
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph and lock must be objects")
    source_commit = graph.get("generated_from_commit")
    if source_commit != lock.get("current_commit") or not _git_commit_exists_antigravity(source_commit):
        raise ValueError("source commit binding is invalid")
    expected_parent = _expected_map_parent_antigravity()
    if expected_parent is not None and source_commit != expected_parent:
        raise ValueError("source commit differs from map parent")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    flows = graph.get("flows")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(flows, list) or len(nodes) > 20:
        raise ValueError("invalid primary node/edge/flow topology")
    node_ids = {node.get("id") for node in nodes if isinstance(node, dict)}
    if len(node_ids) != len(nodes) or None in node_ids:
        raise ValueError("nodes require unique ids")
    def check_evidence(item: object) -> None:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("symbol"), str):
            raise ValueError("missing evidence")
        candidate = ROOT / item["path"]
        if not candidate.is_file() or item["symbol"] not in candidate.read_text(encoding="utf-8"):
            raise ValueError(f"unresolved evidence: {item}")
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("evidence"), list):
            raise ValueError("node requires evidence")
        for item in node["evidence"]:
            check_evidence(item)
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in node_ids or edge.get("to") not in node_ids or edge.get("type") not in EDGE_TYPES:
            raise ValueError("unsupported or unresolved edge")
        check_evidence(edge.get("evidence"))
        if str(edge["from"]).startswith("future_"):
            raise ValueError("no future product-source edge is permitted")
    for flow in flows:
        if not isinstance(flow, dict) or not isinstance(flow.get("steps"), list) or any(step not in node_ids for step in flow["steps"]):
            raise ValueError("unresolved flow reference")
    match = re.search(r'<script id="codemap-data" type="application/json">(.*?)</script>', html)
    if not match or strict_json_antigravity(match.group(1)) != graph:
        raise ValueError("HTML does not embed the exact JSON graph")
    if lock.get("module_fingerprints") != fingerprint_modules_antigravity():
        raise ValueError("fingerprint drift")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--generated-at")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(strict_json_antigravity(JSON_PATH.read_text(encoding="utf-8")), strict_json_antigravity(LOCK_PATH.read_text(encoding="utf-8")), HTML_PATH.read_text(encoding="utf-8"))
        return
    if not args.source_commit:
        parser.error("--source-commit is required unless --validate-only is used")
    generated_at = args.generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    graph, lock, html = build_map_antigravity(args.source_commit, generated_at)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HTML_PATH.write_text(html + "\n", encoding="utf-8")
    validate_map_antigravity(graph, lock, html)


if __name__ == "__main__":
    main_antigravity()
