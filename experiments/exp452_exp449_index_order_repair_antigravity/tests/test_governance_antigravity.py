"""Governance regression for the planned, label-free Exp452 repair."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP452_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP452_RUNBOOK_antigravity_KO.html",
    "docs/EXP452_IMPLEMENTATION_REPORT_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp452_antigravity.html",
    "docs/codemap/codemap_exp452_antigravity.json",
    "docs/codemap/codemap_exp452_antigravity.lock",
)


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


def map_builder_antigravity() -> object:
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp452_antigravity.py"
    spec = importlib.util.spec_from_file_location("exp452_map_builder_antigravity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp452 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GovernanceAntigravityTest(unittest.TestCase):
    def test_governance_artifacts_exist_as_html_and_plan_tracks_completed_tasks_antigravity(self) -> None:
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            with self.subTest(relative=relative):
                self.assertIn("antigravity", Path(relative).name)
                self.assertTrue((ROOT_ANTIGRAVITY / relative).is_file())
        for relative in DOCUMENTS_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertEqual(path.suffix, ".html")
            self.assertFalse(path.with_suffix(".md").exists())
            self.assertIn("<!doctype html>", path.read_text(encoding="utf-8").lower())
        plan = (ROOT_ANTIGRAVITY / "docs/superpowers/plans/2026-08-13-exp452-exp449-index-order-repair-antigravity.html").read_text(encoding="utf-8")
        completed_sections: list[tuple[int, int]] = []
        for task_number, expected_tags in ((1, 7), (2, 8), (3, 8), (4, 9), (5, 9)):
            section = re.search(rf"<section><h2><input\b[^>]*> Task {task_number}:.*?</section>", plan, flags=re.IGNORECASE | re.DOTALL)
            self.assertIsNotNone(section)
            assert section is not None
            completed_tags = re.findall(r"<input\b[^>]*>", section.group(0), flags=re.IGNORECASE)
            self.assertEqual(len(completed_tags), expected_tags)
            for tag in completed_tags:
                self.assertRegex(tag, r"\bchecked(?:\s|=|>)")
            completed_sections.append((section.start(), section.end()))
        task_six = re.search(r"<section><h2><input\b[^>]*> Task 6:.*?</section>", plan, flags=re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(task_six)
        assert task_six is not None
        task_six_tags = re.findall(r"<input\b[^>]*>", task_six.group(0), flags=re.IGNORECASE)
        self.assertEqual(len(task_six_tags), 9)
        for index in (1, 2, 3, 7):
            self.assertRegex(task_six_tags[index], r"\bchecked(?:\s|=|>)")
        for index in (0, 4, 5, 6, 8):
            self.assertNotRegex(task_six_tags[index], r"\bchecked(?:\s|=|>)")

    def test_checked_plan_input_fixture_is_detected_antigravity(self) -> None:
        checked_tag = '<input disabled type="checkbox" checked>'
        self.assertRegex(checked_tag, r"\bchecked(?:\s|=|>)")

    def test_implementation_candidate_truth_is_label_free_and_non_promoting_antigravity(self) -> None:
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        self.assertIsInstance(state, dict)
        self.assertIsInstance(catalog, dict)
        self.assertIsInstance(champion, dict)
        exp452 = state["exp452"]
        lifecycle = {"readiness_invocations": 1, "evaluate_invocations": 0, "inspect_invocations": 0, "label_sessions": 0, "label_query_count": 0, "label_row_count": 0, "readiness_exit_code": 1, "evaluate_exit_code": None, "inspect_exit_code": None}
        self.assertEqual(exp452["status"], "official_readiness_failed_immutable_no_retry")
        self.assertEqual(exp452["label_access_count"], 0)
        self.assertFalse(exp452["test_labels_opened"])
        self.assertTrue(exp452["official_readiness_started"])
        self.assertFalse(exp452["official_evaluate_started"])
        self.assertFalse(exp452["official_inspect_started"])
        self.assertEqual(exp452["official_lifecycle"], lifecycle)
        self.assertIsNone(exp452["prediction_sha256"])
        self.assertIsNone(exp452["result_sha256"])
        self.assertIsNone(exp452["evaluation_sha256"])
        self.assertIsNone(exp452["decision_sha256"])
        self.assertIsNone(exp452["inspect_sha256"])
        self.assertFalse(exp452["promotion_allowed"])
        self.assertFalse(exp452["routing_allowed"])
        self.assertFalse(exp452["pending_candidate_changed"])
        self.assertEqual(exp452["official_failure"]["phase"], "readiness")
        self.assertFalse(exp452["official_failure"]["retry_permitted"])
        self.assertIn("qualified", exp452["qualified_two_boundary"].lower())
        self.assertIn("cannot guide", exp452["qualified_two_boundary"].lower())
        catalog_entry = next(entry for entry in catalog["experiments"] if entry["id"] == "Exp452")
        self.assertEqual(catalog_entry["status"], "official_readiness_failed_immutable_no_retry")
        self.assertEqual(catalog_entry["official_lifecycle"], lifecycle)
        self.assertIsNone(catalog_entry["prediction_sha256"])
        self.assertIsNone(catalog_entry["result_sha256"])
        self.assertIsNone(catalog_entry["evaluation_sha256"])
        self.assertIsNone(catalog_entry["decision_sha256"])
        self.assertIsNone(catalog_entry["inspect_sha256"])
        self.assertFalse(catalog_entry["promotion_allowed"])
        self.assertFalse(catalog_entry["routing_allowed"])
        self.assertFalse(catalog_entry["pending_candidate_changed"])
        self.assertEqual(catalog_entry["official_failure"]["phase"], "readiness")
        self.assertFalse(catalog_entry["official_failure"]["retry_permitted"])
        champion_entry = next(entry for entry in champion["history"] if entry["experiment"] == "Exp452")
        self.assertEqual(champion_entry["status"], "official_readiness_failed_immutable_no_retry")
        self.assertEqual(champion_entry["official_lifecycle"], lifecycle)
        self.assertIsNone(champion_entry["prediction_sha256"])
        self.assertIsNone(champion_entry["result_sha256"])
        self.assertIsNone(champion_entry["evaluation_sha256"])
        self.assertIsNone(champion_entry["decision_sha256"])
        self.assertIsNone(champion_entry["inspect_sha256"])
        self.assertFalse(champion_entry["promotion_allowed"])
        self.assertFalse(champion_entry["routing_allowed"])
        self.assertFalse(champion_entry["pending_candidate_changed"])
        self.assertEqual(champion_entry["official_failure"]["phase"], "readiness")
        self.assertFalse(champion_entry["official_failure"]["retry_permitted"])
        self.assertFalse((ROOT_ANTIGRAVITY / "EXPERIMENT_HISTORY.json").exists())

    def test_map_is_strict_embedded_and_records_task_five_cli_antigravity(self) -> None:
        builder = map_builder_antigravity()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        html = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        self.assertIsInstance(graph, dict)
        self.assertLessEqual(len(graph["nodes"]), 20)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertIn("task2_regressions", node_by_id)
        self.assertIn("task5_cli", node_by_id)
        self.assertFalse(graph["unknown_relationships"])
        self.assertTrue(any(edge["from"] == "task5_cli" and edge["to"] == "task3_readiness" and edge["type"] == "reads" for edge in graph["edges"]))
        self.assertTrue(any(edge["from"] == "task5_cli" and edge["to"] == "task4_evaluator" and edge["type"] == "reads" for edge in graph["edges"]))
        self.assertTrue(any(edge["from"] == "task5_cli" and edge["to"] == "task4_replay" and edge["type"] == "reads" for edge in graph["edges"]))
        builder.validate_map_antigravity(graph, lock, html)

    def test_map_answers_task_two_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: leaving implemented Task 2 foundations hidden behind a future/unknown node."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        expected_tests = {
            "task2_config": "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_config_antigravity.py",
            "task2_contracts": "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_contracts_antigravity.py",
            "task2_artifacts": "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_artifacts_antigravity.py",
            "task2_bootstrap": "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_bootstrap_antigravity.py",
            "task2_dashboard": "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_dashboard_export_antigravity.py",
        }
        differential = "experiments/exp452_exp449_index_order_repair_antigravity/tests/test_controlled_fork_differential_antigravity.py"
        for node_id, focused_test in expected_tests.items():
            with self.subTest(node_id=node_id):
                self.assertIn(node_id, node_by_id)
                self.assertNotEqual(node_by_id[node_id].get("status"), "future_unknown")
                self.assertEqual(node_by_id[node_id]["tests"], [focused_test, differential])
                self.assertTrue(any(edge["from"] == "task2_regressions" and edge["to"] == node_id and edge["type"] == "tests" for edge in graph["edges"]))

    def test_map_answers_task_three_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: implemented predecessor/label/lifecycle/readiness remain falsely unknown."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        expected = {
            "task3_predecessor": "test_predecessor_antigravity.py",
            "task3_sources": "test_sources_antigravity.py",
            "task3_lifecycle": "test_lifecycle_antigravity.py",
            "task3_readiness": "test_readiness_antigravity.py",
        }
        for node_id, test_leaf in expected.items():
            with self.subTest(node_id=node_id):
                self.assertIn(node_id, node_by_id)
                self.assertNotEqual(node_by_id[node_id].get("status"), "future_unknown")
                self.assertTrue(any(path.endswith(test_leaf) for path in node_by_id[node_id]["tests"]))
                self.assertIn("caller", node_by_id[node_id]["role"].lower())
                self.assertIn("effect", node_by_id[node_id]["role"].lower())
        self.assertTrue(any(edge["from"] == "task3_sources" and edge["to"] == "task3_predecessor" and edge["type"] == "reads" for edge in graph["edges"]))
        self.assertTrue(any(edge["from"] == "task3_readiness" and edge["to"] == "task3_predecessor" and edge["type"] == "reads" for edge in graph["edges"]))
        self.assertTrue(any(edge["from"] == "task3_readiness" and edge["to"] == "task3_lifecycle" and edge["type"] == "reads" for edge in graph["edges"]))

    def test_documents_exclude_exp449_qualified_two_as_a_decision_input_antigravity(self) -> None:
        for relative in DOCUMENTS_ANTIGRAVITY:
            content = (ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8").lower()
            self.assertIn("qualified", content)
            self.assertIn("cannot guide", content)


if __name__ == "__main__":
    unittest.main()
