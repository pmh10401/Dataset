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
    "5041bc34c5323658779fbf017925dec80be5e315efd56f5189c94f7c9dcb36b2"
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
        self.assertEqual(state["exp451"]["status"], "planned_not_started")
        self.assertEqual(
            protected_exp449_state_sha256_antigravity(state),
            PROTECTED_EXP449_SHA256_ANTIGRAVITY,
        )
        self.assertEqual(state["exp451"]["label_access_count"], 0)
        self.assertFalse(state["exp451"]["official_readiness_started"])
        self.assertEqual(state["exp451"]["target_dataset_count"], 48)
        self.assertEqual(state["exp451"]["target_test_rows"], 72_300)

    def test_exp451_governance_documents_are_self_contained_antigravity(self) -> None:
        for relative_path in (
            "docs/EXP451_PREREGISTRATION_antigravity_KO.html",
            "docs/EXP451_RUNBOOK_antigravity_KO.html",
            "docs/EXP451_IMPLEMENTATION_REPORT_antigravity.html",
        ):
            content = (ROOT_ANTIGRAVITY / relative_path).read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", content.lower())
            self.assertIn("Exp451", content)

    def test_map_rejects_unknown_relationship_with_ghost_node_ids_antigravity(self) -> None:
        builder, graph, lock, rendered_html = valid_map_triplet_antigravity()
        graph = copy.deepcopy(graph)
        graph["unknown_relationships"].append({"from": "ghost", "to": "ghost", "reason": "invalid reference"})
        rendered_html = builder._self_contained_html_antigravity(graph)
        with self.assertRaisesRegex(ValueError, "invalid unknown reference"):
            builder.validate_map_antigravity(graph, lock, rendered_html)

    def test_map_rejects_known_edge_to_future_unknown_node_antigravity(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "unmarked unknown relationship"):
            builder.validate_map_antigravity(graph, lock, rendered_html)

    def test_map_rejects_existing_but_nonfunctional_parent_commit_antigravity(self) -> None:
        builder = map_builder_module_antigravity()
        graph, lock, rendered_html = builder.build_map_antigravity(BASE_COMMIT_ANTIGRAVITY, "2026-08-13T00:00:00Z")
        lock["working_tree_dirty"] = False
        with self.assertRaisesRegex(ValueError, "invalid source parent relationship"):
            builder.validate_map_antigravity(graph, lock, rendered_html)


if __name__ == "__main__":
    unittest.main()
