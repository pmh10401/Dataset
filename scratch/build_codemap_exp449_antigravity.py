#!/usr/bin/env python3
"""Build and validate the Exp449 map after the approved two-root correction plan."""

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
_MAP_TRIPLET_ANTIGRAVITY = {
    "docs/codemap/codemap_exp449_antigravity.html",
    "docs/codemap/codemap_exp449_antigravity.json",
    "docs/codemap/codemap_exp449_antigravity.lock",
}


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
        if any(part in EXCLUDED_PARTS for part in parts) or relative.as_posix() in _MAP_TRIPLET_ANTIGRAVITY:
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


def _changed_paths_from_parent_antigravity(source_parent: str) -> list[str]:
    """List represented tracked changes from the declared same-commit parent."""
    return sorted(path for path in subprocess.check_output(["git", "diff", "--name-only", source_parent], cwd=ROOT, text=True).splitlines() if path)


def build_map_antigravity(source_parent: str, generated_at: str) -> tuple[dict[str, object], dict[str, object], str]:
    """Return JSON graph, lock, and self-contained HTML for the same graph."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_parent):
        raise ValueError("source_parent must be lowercase 40-hex")
    if not _git_commit_exists_antigravity(source_parent):
        raise ValueError("source parent does not exist")
    nodes: list[dict[str, object]] = [
        {"id": "research_control", "path": "RESEARCH_STATE.json", "role": "Planned Exp449 state and consolidated non-promotion truth: zero label access, closed official lifecycle, and unchanged champion/pending candidate.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("RESEARCH_STATE.json", '"exp449"'), evidence_antigravity("CHAMPION_LEDGER.json", '"experiment": "Exp449"')]},
        {"id": "exp449_preregistration", "path": "docs/EXP449_PREREGISTRATION_antigravity_KO.html", "role": "Frozen candidates, settings, success gates, label boundary, CPU policy, and no-promotion contract.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "official_label_sessions = 1")]},
        {"id": "exp449_runbook", "path": "docs/EXP449_RUNBOOK_antigravity_KO.html", "role": "Single-use future command order, seven thread variables, stop rules, and dashboard firewall.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/EXP449_RUNBOOK_antigravity_KO.html", "AEON_NUM_THREADS=1")]},
        {"id": "project_execution_policy", "path": "AGENTS.md", "role": "Durable global rule: checked HTML-plan controls represent only completed implementation and review/verification boundaries.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("AGENTS.md", "In HTML implementation plans")]},
        {"id": "two_root_correction_design", "path": "docs/superpowers/specs/2026-08-12-exp449-two-root-lifecycle-correction-antigravity-design.html", "role": "Approved receipt-backed correction authority that separates immutable prediction authority from evaluation publication and makes lifecycle progress durable.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/superpowers/specs/2026-08-12-exp449-two-root-lifecycle-correction-antigravity-design.html", "Exp449 two-root lifecycle receipt correction")]},
        {"id": "two_root_correction_plan", "path": "docs/superpowers/plans/2026-08-12-exp449-two-root-lifecycle-correction-antigravity.html", "role": "Executable superseding plan: original Tasks 1–6 remain checked, original Tasks 7–8 remain unchecked, and prospective receipt-backed two-root work uses same-commit maps.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("docs/superpowers/plans/2026-08-12-exp449-two-root-lifecycle-correction-antigravity.html", "Task 5: Authorized single-use lifecycle and truth/dashboard closeout"), evidence_antigravity("docs/superpowers/plans/2026-08-12-exp449-residual-fn-method-qualification-antigravity.html", "Task 6: One-Session Counterfactual Evaluation and Source-Free Replay")]},
        {"id": "governance_regression", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "role": "Structural stdlib HTML-parser regression for Exp449 plan task/step completion state and global governance constraints.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "class PlanCheckboxParserAntigravity")]},
        {"id": "candidate_registry", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py", "role": "Frozen eight-family, two-candidate registry and pinned environment authority.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_config_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY")]},
        {"id": "immutable_contracts", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py", "role": "Frozen slots exact-type authorities for candidates, audits, bundles, baselines, and metrics.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_contracts_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/contracts_antigravity.py", "class MethodAuditAntigravity")]},
        {"id": "canonical_artifacts", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py", "role": "Canonical JSON/JSONL and create-only no-follow artifact inventory boundary used by immutable predecessor capture.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_artifacts_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/artifacts_antigravity.py", "def artifact_map_bytes_antigravity")]},
        {"id": "method_audit", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "role": "Subprocess-only fixed-interpreter audit binding versions, source SHA256 values, and frozen public methods.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_method_audit_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "def audit_methods_antigravity")]},
        {"id": "exp448_authority", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "role": "Descriptor-pinned immutable Exp448 11-leaf authority, independently binding the source-free Exp446 stable-26 publication.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_predecessor_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "def capture_exp448_authority_antigravity")]},
        {"id": "label_free_source", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py", "role": "Read-only descriptor-pinned value loader and Task-6-only closure-private TEST-label session boundary. It consumes Task-5's exact authority validator before DB reachability.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_sources_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py", "def _frozen_prediction_order_antigravity")]},
        {"id": "representation_sources", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py", "role": "Four deterministic fixed aeon 1.4.0 CPU representation adapters, including Hydra-first scaled hybrid concatenation and no classifier head.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_representations_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/representations_antigravity.py", "def make_representation_antigravity")]},
        {"id": "score_sources", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "role": "Sequential deterministic five-fold, reference-capped, chunked label-free scores; its frozen score contract is the future Task-5 prediction input.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_scoring_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "def score_candidate_antigravity")]},
        {"id": "acceleration_gate", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/mlx_gate_antigravity.py", "role": "Stage-specific immutable CPU/MLX/MPS parity and speed gate; absent MLX in the fixed environment retains CPU canonical scoring.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_mlx_gate_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/mlx_gate_antigravity.py", "def assess_mlx_antigravity")]},
        {"id": "receipt_lifecycle", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/lifecycle_antigravity.py", "role": "Task-2 descriptor-safe receipt state machine and unified create-only failure audit. Classification registers every leaf descriptor before validation; root constructors close partial authority; complete precommit closes root and audit in close-all order. Command root/audit authority stays live through complete reread, exact-inode cleanup removes an undurable leaf before recovery, and only receipt directory then parent fsync create success. Official execution remains closed.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_lifecycle_antigravity.py", "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_prediction_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/lifecycle_antigravity.py", "class _LifecycleTransactionAntigravity"), evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/lifecycle_antigravity.py", "class _RootIdentityTransactionAntigravity"), evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/lifecycle_antigravity.py", "def _close_complete_authorities_antigravity")]},
        {"id": "prediction_authority", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "role": "Task-2 receipt-gated public predict wrapper around the frozen label-free 96-unit, 11-worker prediction authority and immutable descriptor validator; it imports prediction-owned exact-prefix progress commits whose 96 durable leaves gate completion.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_prediction_antigravity.py", "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_progress_antigravity.py", "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_lifecycle_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "def predict_antigravity"), evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "progress_antigravity import"), evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/progress_antigravity.py", "def commit_candidate_leaf_antigravity")]},
        {"id": "task6_evaluation", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py", "role": "Task-6 one-session synthetic counterfactual evaluator with fixed PCG64 paired bootstrap, exact Task-5 descriptor capture, retained-FD post-rename identity/replay recovery, and phase/recovery-private audit evidence for its 12-leaf publication.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_evaluation_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py", "def _evaluate_with_dependencies_antigravity")]},
        {"id": "task6_replay", "path": "experiments/exp449_residual_fn_method_qualification_antigravity/replay_antigravity.py", "role": "Task-6 descriptor-pinned source-free canonical semantic replay of twelve leaves against actual Task-5 authority, canonical rows, inventory, leaf/aggregate, context/source, and manifest provenance. Bare inspect enters lifecycle before receipt-derived commit/capture/replay.", "tests": ["experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_replay_antigravity.py", "experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_lifecycle_antigravity.py"], "evidence": [evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/replay_antigravity.py", "def _inspect_with_dependencies_antigravity")]},
    ]
    edges: list[dict[str, object]] = [
        {"from": "research_control", "to": "exp449_preregistration", "type": "reads", "evidence": evidence_antigravity("RESEARCH_STATE.json", "EXP449_PREREGISTRATION_antigravity_KO.html")},
        {"from": "research_control", "to": "exp449_runbook", "type": "reads", "evidence": evidence_antigravity("RESEARCH_STATE.json", "EXP449_RUNBOOK_antigravity_KO.html")},
        {"from": "project_execution_policy", "to": "two_root_correction_plan", "type": "governs", "evidence": evidence_antigravity("AGENTS.md", "In HTML implementation plans")},
        {"from": "two_root_correction_design", "to": "two_root_correction_plan", "type": "governs", "evidence": evidence_antigravity("docs/superpowers/specs/2026-08-12-exp449-two-root-lifecycle-correction-antigravity-design.html", "The lifecycle has independent immutable prediction")},
        {"from": "governance_regression", "to": "project_execution_policy", "type": "reads", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "Path(\"AGENTS.md\")")},
        {"from": "governance_regression", "to": "two_root_correction_plan", "type": "reads", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/tests/test_governance_antigravity.py", "test_html_plan_records_only_verified_task_and_step_progress_antigravity")},
        {"from": "exp449_preregistration", "to": "candidate_registry", "type": "governs", "evidence": evidence_antigravity("docs/EXP449_PREREGISTRATION_antigravity_KO.html", "global_fp_max = 314")},
        {"from": "method_audit", "to": "candidate_registry", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "config_antigravity import")},
        {"from": "method_audit", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/method_audit_antigravity.py", "contracts_antigravity import")},
        {"from": "exp448_authority", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "contracts_antigravity import")},
        {"from": "exp448_authority", "to": "canonical_artifacts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "artifacts_antigravity import")},
        {"from": "exp448_authority", "to": "candidate_registry", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/predecessor_antigravity.py", "config_antigravity import")},
        {"from": "label_free_source", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/sources_antigravity.py", "contracts_antigravity import")},
        {"from": "candidate_registry", "to": "representation_sources", "type": "governs", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/config_antigravity.py", "minirocket_requested_kernels")},
        {"from": "score_sources", "to": "representation_sources", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "representations_antigravity import")},
        {"from": "score_sources", "to": "immutable_contracts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/scoring_antigravity.py", "contracts_antigravity import")},
        {"from": "prediction_authority", "to": "receipt_lifecycle", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "lifecycle_antigravity import")},
        {"from": "receipt_lifecycle", "to": "canonical_artifacts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/lifecycle_antigravity.py", "artifacts_antigravity import")},
        {"from": "prediction_authority", "to": "score_sources", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "empirical_candidates")},
        {"from": "prediction_authority", "to": "score_sources", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/prediction_antigravity.py", "from .scoring_antigravity import score_candidate_antigravity")},
        {"from": "task6_evaluation", "to": "exp448_authority", "type": "reads", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py", "capture_exp448_antigravity")},
        {"from": "task6_evaluation", "to": "receipt_lifecycle", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py", "lifecycle_antigravity import")},
        {"from": "task6_evaluation", "to": "prediction_authority", "type": "reads", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py", "capture_frozen_prediction_authority_antigravity")},
        {"from": "task6_evaluation", "to": "label_free_source", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py", "sources_antigravity import")},
        {"from": "task6_evaluation", "to": "task6_replay", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/evaluation_antigravity.py", "replay_antigravity import")},
        {"from": "task6_replay", "to": "canonical_artifacts", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/replay_antigravity.py", "artifacts_antigravity import")},
        {"from": "task6_replay", "to": "receipt_lifecycle", "type": "imports", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/replay_antigravity.py", "lifecycle_antigravity import")},
        {"from": "task6_replay", "to": "prediction_authority", "type": "reads", "evidence": evidence_antigravity("experiments/exp449_residual_fn_method_qualification_antigravity/replay_antigravity.py", "capture_frozen_prediction_authority_antigravity")},
    ]
    represented_paths = sorted(set(_changed_paths_from_parent_antigravity(source_parent)) | _MAP_TRIPLET_ANTIGRAVITY)
    graph: dict[str, object] = {"generated_at": generated_at, "source_parent_commit": source_parent, "scope": ["Task-1 correction is complete and Task-2 receipt lifecycle is implemented for review; readiness/CLI, official lifecycle execution, truth closeout, dashboard access, and publication remain unauthorized."], "topology_note": "The map retains 20 nodes by consolidating prediction progress into the public prediction-authority boundary while retaining its concrete import, progress tests, 96-leaf completion gate, and lifecycle flow. The lock uses the same-commit source-parent contract and excludes exactly its own three Exp449 triplet paths.", "nodes": nodes, "edges": edges, "flows": [{"trigger": "Receipt-backed lifecycle", "steps": ["two_root_correction_design", "receipt_lifecycle", "prediction_authority", "task6_evaluation", "task6_replay"], "outcome": "Task-2 provides receipt/audit mechanics and source-free replay wrappers without authorizing any official command; prediction_authority includes its exact-prefix progress dependency and 96-leaf durable completion gate."}, {"trigger": "Descriptor capture to sole label session", "steps": ["prediction_authority", "receipt_lifecycle", "task6_evaluation", "label_free_source", "task6_replay"], "outcome": "Evaluation captures the immutable prediction authority before evaluate-started, opens exactly one 48-query/72,300-row label session only after that receipt, and publishes exactly twelve evaluation leaves before evaluate-complete."}, {"trigger": "Task-4 label-free score generation", "steps": ["label_free_source", "representation_sources", "score_sources", "acceleration_gate"], "outcome": "CPU aeon representations and Exp249 LocalGap return finite deterministic scores; MLX/MPS availability cannot approve acceleration without the full gate."}], "unknown_relationships": ["Same-commit readiness/CLI validation, clean-candidate truth, single-use official commands, and presentation-only dashboard publication remain future plan stages."], "unknowns": ["Exp448 locally binds Sites source commit 500ccbe12b07475e51e8cdcbfa8619d9fbdefddb, version 6, and one-account/no-group custom owner ACL; no local dashboard URL exists, dashboard_access remains blocked, and future Task 4 must obtain it only through authenticated metadata reads."]}
    lock: dict[str, object] = {"schema": "exp449_codemap_same_commit_v1", "source_parent_commit": source_parent, "working_tree_dirty": False, "represented_paths": represented_paths, "generated_at": generated_at, "scanned_scope": ["tracked root files including AGENTS.md", *SCANNED_TOP_LEVEL], "excluded_directories": sorted(EXCLUDED_PARTS), "excluded_paths": sorted(_MAP_TRIPLET_ANTIGRAVITY), "fingerprint_algorithm": "sha256-v1: proposed post-change tracked tree; sort every tracked repo-relative path except the exact Exp449 map triplet by module; append UTF-8 path, NUL, decimal byte length, NUL, current bytes, NUL.", "module_fingerprints": fingerprint_modules_antigravity(), "changed_modules_since_source_parent": sorted({Path(path).parts[0] if len(Path(path).parts) > 1 else "root" for path in represented_paths})}
    payload = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    html = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Exp449 Code Map</title><style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px/1.55 system-ui}pre{white-space:pre-wrap;overflow:auto;background:#eef2f7;padding:1rem}</style></head><body><main><h1>Exp449 two-root correction code map</h1><p>Task 2 implementation is represented for review; official lifecycle, dashboard access, and publication remain closed. This map is evidence only and does not authorize a report or dashboard action.</p><script id=\"codemap-data\" type=\"application/json\">" + payload + "</script><pre id=\"graph\"></pre></main><script>const data=JSON.parse(document.getElementById('codemap-data').textContent);document.getElementById('graph').textContent=JSON.stringify(data,null,2);</script></body></html>"
    return graph, lock, html


def validate_map_antigravity(graph: object, lock: object, html: str, *, precommit: bool = False) -> None:
    """Reject missing evidence, unresolved refs, >20 primary nodes, and fingerprint drift."""
    if not isinstance(graph, dict) or not isinstance(lock, dict):
        raise ValueError("graph and lock must be objects")
    source_parent = lock.get("source_parent_commit")
    # Historical in-memory callers may still supply the former explicit map
    # provenance fields.  Validate them before cleanliness so malformed
    # provenance is never masked by an unrelated dirty candidate tree.
    legacy_source, legacy_current = graph.get("generated_from_commit"), lock.get("current_commit")
    if legacy_source is not None or legacy_current is not None:
        if not _git_commit_exists_antigravity(legacy_source) or not _git_commit_exists_antigravity(legacy_current):
            raise ValueError("source commit is invalid")
        if subprocess.check_output(["git", "rev-parse", f"{legacy_current}^"], cwd=ROOT, text=True).strip() != legacy_source:
            raise ValueError("map parent is invalid")
    if lock.get("schema") != "exp449_codemap_same_commit_v1" or graph.get("source_parent_commit") != source_parent or not _git_commit_exists_antigravity(source_parent):
        raise ValueError("same-commit source-parent binding is invalid")
    represented_paths = lock.get("represented_paths")
    if not isinstance(represented_paths, list) or any(not isinstance(path, str) for path in represented_paths) or set(_MAP_TRIPLET_ANTIGRAVITY) - set(represented_paths):
        raise ValueError("same-commit represented paths are invalid")
    actual_paths = _changed_paths_from_parent_antigravity(source_parent)
    if sorted(represented_paths) != actual_paths:
        raise ValueError("unrepresented or stale same-commit paths")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if precommit:
        head_parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT, text=True).strip()
        if head != source_parent and head_parent != source_parent:
            raise ValueError("precommit generation must name checked-out parent or amendable first parent")
    else:
        if subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT, text=True).strip() != source_parent or subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False).stdout:
            raise ValueError("reviewed same-commit map must be clean and parent-bound")
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
    if lock.get("working_tree_dirty") is not False or lock.get("module_fingerprints") != fingerprint_modules_antigravity():
        raise ValueError("fingerprint drift")


def main_antigravity() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parent")
    parser.add_argument("--generated-at")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_map_antigravity(strict_json_antigravity(JSON_PATH.read_text(encoding="utf-8")), strict_json_antigravity(LOCK_PATH.read_text(encoding="utf-8")), HTML_PATH.read_text(encoding="utf-8"))
        return
    if not args.source_parent:
        parser.error("--source-parent is required unless --validate-only is used")
    generated_at = args.generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    graph, lock, html = build_map_antigravity(args.source_parent, generated_at)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOCK_PATH.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HTML_PATH.write_text(html + "\n", encoding="utf-8")
    validate_map_antigravity(graph, lock, html, precommit=True)


if __name__ == "__main__":
    main_antigravity()
