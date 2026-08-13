"""Governance regression for the planned, label-free Exp451 sibling."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PROTECTED_EXP449_KEYS_ANTIGRAVITY = (
    "active_experiment",
    "active_phase",
    "status",
    "active_documents",
    "pending_candidate",
    "exp449",
    "next_action",
    "exact_next_command",
    "official_readiness_command_after_fresh_sol_ship",
    "approval_scope",
)
PROTECTED_EXP449_SHA256_ANTIGRAVITY = (
    "e8e1114bc0c837a68f83464f4f92ddcb08945d95a67f09ebb1dea3d82c0886aa"
)
BASE_COMMIT_ANTIGRAVITY = "b5a3623c13d6db241a9b83e36b7e1f3937caddfe"


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


def protected_exp449_state_sha256_antigravity(state: dict[str, object]) -> str:
    payload = {key: state[key] for key in PROTECTED_EXP449_KEYS_ANTIGRAVITY}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    return hashlib.sha256(data).hexdigest()


def map_builder_module_antigravity() -> object:
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp451_antigravity.py"
    spec = importlib.util.spec_from_file_location("exp451_map_builder_antigravity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp451 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_source_commit_antigravity() -> str:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
    changed = set(subprocess.check_output(["git", "diff", "--name-only", "HEAD^", "HEAD"], cwd=ROOT_ANTIGRAVITY, text=True).splitlines())
    triplet = {
        "docs/codemap/codemap_exp451_antigravity.html",
        "docs/codemap/codemap_exp451_antigravity.json",
        "docs/codemap/codemap_exp451_antigravity.lock",
    }
    return parent if changed == triplet else head


def valid_map_triplet_antigravity() -> tuple[object, dict[str, object], dict[str, object], str]:
    builder = map_builder_module_antigravity()
    graph, lock, rendered_html = builder.build_map_antigravity(valid_source_commit_antigravity(), "2026-08-13T00:00:00Z")
    lock["working_tree_dirty"] = False
    builder.validate_map_antigravity(graph, lock, rendered_html)
    return builder, graph, lock, rendered_html


class GovernanceAntigravityTest(unittest.TestCase):
    def test_exp451_planned_truth_antigravity(self) -> None:
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        self.assertIsInstance(state, dict)
        self.assertEqual(state["active_experiment"], "Exp449")
        self.assertEqual(state["active_phase"], "official_readiness_passed_evaluation_failed_immutable_no_retry")
        self.assertEqual(state["status"], "official_readiness_passed_evaluation_failed_immutable_no_retry")
        self.assertEqual(state["exp451"]["status"], "planned_not_started")
        self.assertEqual(
            protected_exp449_state_sha256_antigravity(state),
            PROTECTED_EXP449_SHA256_ANTIGRAVITY,
        )
        self.assertEqual(state["exp451"]["label_access_count"], 0)
        self.assertFalse(state["exp451"]["official_readiness_started"])
        self.assertEqual(state["exp451"]["target_dataset_count"], 48)
        self.assertEqual(state["exp451"]["target_test_rows"], 72_300)
        self.assertEqual(state["exp451"]["scientific_input"], "Exp448 stable-26 byte-authoritative aggregate truth")
        self.assertEqual(state["exp451"]["label_access_count"], 0)
        self.assertEqual(state["exp451"]["official_lifecycle"]["label_sessions"], 0)
        exp449 = state["exp449"]
        exp449_lifecycle = exp449["official_lifecycle"]
        self.assertEqual(exp449["status"], "official_readiness_passed_evaluation_failed_immutable_no_retry")
        self.assertEqual(exp449["label_access_count"], 1)
        self.assertTrue(exp449["test_labels_opened"])
        self.assertEqual(exp449_lifecycle["label_sessions"], 1)
        self.assertEqual(exp449_lifecycle["evaluate_label_query_count"], 48)
        self.assertEqual(exp449_lifecycle["evaluate_label_row_count"], 72_300)
        self.assertEqual(exp449_lifecycle["evaluation_root"], "absent")
        self.assertTrue(exp449_lifecycle["permanent_stop"])
        for root_key in ("prediction_root", "evaluation_root", "readiness_receipt", "failure_audit"):
            self.assertFalse(Path(state["exp451"][root_key]).exists(), root_key)

    def test_exp451_governance_documents_are_self_contained_antigravity(self) -> None:
        for relative_path in (
            "docs/EXP451_PREREGISTRATION_antigravity_KO.html",
            "docs/EXP451_RUNBOOK_antigravity_KO.html",
            "docs/EXP451_IMPLEMENTATION_REPORT_antigravity.html",
        ):
            content = (ROOT_ANTIGRAVITY / relative_path).read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", content.lower())
            self.assertIn("Exp451", content)
        runbook = (ROOT_ANTIGRAVITY / "docs/EXP451_RUNBOOK_antigravity_KO.html").read_text(encoding="utf-8")
        implementation_report = (ROOT_ANTIGRAVITY / "docs/EXP451_IMPLEMENTATION_REPORT_antigravity.html").read_text(encoding="utf-8")
        for content in (runbook, implementation_report):
            self.assertIn("Exp449", content)
            self.assertIn("immutable", content)
            self.assertIn("Exp448", content)

    def test_map_rejects_unknown_relationship_with_ghost_node_ids_antigravity(self) -> None:
        builder, graph, lock, rendered_html = valid_map_triplet_antigravity()
        graph = copy.deepcopy(graph)
        graph["unknown_relationships"].append({"from": "ghost", "to": "ghost", "reason": "invalid reference"})
        rendered_html = builder._self_contained_html_antigravity(graph)
        with self.assertRaisesRegex(ValueError, "invalid unknown reference"):
            builder.validate_map_antigravity(graph, lock, rendered_html)

    def test_map_rejects_edge_to_removed_future_node_antigravity(self) -> None:
        builder, graph, lock, rendered_html = valid_map_triplet_antigravity()
        graph = copy.deepcopy(graph)
        graph["edges"].append(
            {
                "from": "research_state",
                "to": "future_rdst_implementation",
                "type": "reads",
                "evidence": {"path": "RESEARCH_STATE.json", "symbol": "EXP451_PREREGISTRATION_antigravity_KO.html"},
            }
        )
        rendered_html = builder._self_contained_html_antigravity(graph)
        with self.assertRaisesRegex(ValueError, "invalid edge reference"):
            builder.validate_map_antigravity(graph, lock, rendered_html)

    def test_map_rejects_existing_but_nonfunctional_parent_commit_antigravity(self) -> None:
        builder = map_builder_module_antigravity()
        graph, lock, rendered_html = builder.build_map_antigravity(BASE_COMMIT_ANTIGRAVITY, "2026-08-13T00:00:00Z")
        lock["working_tree_dirty"] = False
        with self.assertRaisesRegex(ValueError, "invalid source parent relationship"):
            builder.validate_map_antigravity(graph, lock, rendered_html)

    def test_map_replaces_future_rdst_with_task2_source_and_tests_antigravity(self) -> None:
        builder = map_builder_module_antigravity()
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
        graph, lock, rendered_html = builder.build_map_antigravity(source_commit, "2026-08-13T00:00:00Z")
        lock["working_tree_dirty"] = False
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertNotIn("future_rdst_implementation", node_by_id)
        expected = {
            "rdst_config": ("experiments/exp451_rdst_local_gap_qualification_antigravity/config_antigravity.py", "FROZEN_CONFIG_ANTIGRAVITY", "experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_config_antigravity.py"),
            "rdst_contracts": ("experiments/exp451_rdst_local_gap_qualification_antigravity/contracts_antigravity.py", "FrozenPredictionAuthorityAntigravity", "experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_contracts_antigravity.py"),
            "rdst_method_audit": ("experiments/exp451_rdst_local_gap_qualification_antigravity/method_audit_antigravity.py", "audit_rdst_antigravity", "experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_method_audit_antigravity.py"),
        }
        for node_id, (path, symbol, test) in expected.items():
            with self.subTest(node_id=node_id):
                self.assertEqual(node_by_id[node_id]["path"], path)
                self.assertIn(test, node_by_id[node_id]["tests"])
                self.assertIn({"path": path, "symbol": symbol}, node_by_id[node_id]["evidence"])
        self.assertEqual(graph["unknown_relationships"], [])
        edges = {(edge["from"], edge["to"]): edge for edge in graph["edges"]}
        self.assertNotIn(("rdst_config", "rdst_contracts"), edges)
        self.assertEqual(edges[("approved_design", "rdst_contracts")]["type"], "governs")
        builder.validate_map_antigravity(graph, lock, rendered_html)

    def test_map_connects_task3_rdst_localgap_source_evidence_antigravity(self) -> None:
        builder = map_builder_module_antigravity()
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_ANTIGRAVITY, text=True).strip()
        graph, lock, rendered_html = builder.build_map_antigravity(source_commit, "2026-08-13T00:00:00Z")
        lock["working_tree_dirty"] = False
        nodes = {node["id"]: node for node in graph["nodes"]}
        expected = {
            "official_rdst_representation": ("experiments/exp451_rdst_local_gap_qualification_antigravity/rdst_antigravity.py", "RDSTRepresentationAntigravity", "experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_rdst_antigravity.py"),
            "localgap_scoring": ("experiments/exp451_rdst_local_gap_qualification_antigravity/scoring_antigravity.py", "score_candidate_antigravity", "experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_scoring_antigravity.py"),
            "localgap_backend_gate": ("experiments/exp451_rdst_local_gap_qualification_antigravity/backend_gate_antigravity.py", "decide_backend_antigravity", "experiments/exp451_rdst_local_gap_qualification_antigravity/tests/test_backend_gate_antigravity.py"),
        }
        for node_id, (path, symbol, test) in expected.items():
            with self.subTest(node_id=node_id):
                self.assertEqual(nodes[node_id]["path"], path)
                self.assertIn(test, nodes[node_id]["tests"])
                self.assertIn({"path": path, "symbol": symbol}, nodes[node_id]["evidence"])


if __name__ == "__main__":
    unittest.main()
