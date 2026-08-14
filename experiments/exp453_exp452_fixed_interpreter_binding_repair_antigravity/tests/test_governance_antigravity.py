"""Governance contract for the verified, still label-free Exp453 candidate."""

from __future__ import annotations

import hashlib
import html as html_module
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PLAN_ANTIGRAVITY = "docs/superpowers/plans/2026-08-13-exp453-fixed-interpreter-binding-repair-antigravity.html"
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP453_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP453_RUNBOOK_antigravity_KO.html",
    "docs/EXP453_IMPLEMENTATION_REPORT_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp453_antigravity.html",
    "docs/codemap/codemap_exp453_antigravity.json",
    "docs/codemap/codemap_exp453_antigravity.lock",
)
FIXED_EXECUTABLE_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
FIXED_PREFIX_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
PREDICTION_SHA256_ANTIGRAVITY = "a6054a44fc4e78f354dcf61d2e003284804680cd4197405364fcb5ef8d822e0d"


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


def map_builder_antigravity() -> object:
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp453_antigravity.py"
    spec = importlib.util.spec_from_file_location("exp453_map_builder_antigravity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp453 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GovernanceAntigravityTest(unittest.TestCase):
    def test_task_one_html_and_checkbox_progress_antigravity(self) -> None:
        """Break caught: Task 1 is called complete while its HTML or checkboxes are missing."""
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            with self.subTest(relative=relative):
                path = ROOT_ANTIGRAVITY / relative
                self.assertIn("antigravity", path.name)
                self.assertTrue(path.is_file())
        for relative in DOCUMENTS_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertEqual(path.suffix, ".html")
            self.assertFalse(path.with_suffix(".md").exists())
            self.assertIn("<!doctype html>", path.read_text(encoding="utf-8").lower())

        plan = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        for task_number in range(1, 7):
            section = re.search(
                rf"<section><h2><input\b[^>]*> Task {task_number}:.*?</section>",
                plan,
                flags=re.IGNORECASE | re.DOTALL,
            )
            self.assertIsNotNone(section)
            assert section is not None
            tags = re.findall(r"<input\b[^>]*>", section.group(0), flags=re.IGNORECASE)
            checked = [tag for tag in tags if re.search(r"\bchecked(?:\s|=|>)", tag)]
            if task_number in (1, 2, 3, 4, 5):
                self.assertEqual(len(checked), len(tags))
            else:
                self.assertEqual(len(tags), 9)
                self.assertEqual(len(checked), 7)

    def test_verified_candidate_truth_preserves_science_and_champion_antigravity(self) -> None:
        """Break caught: verified implementation truth claims an official result or changes science."""
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)
        exp453 = state["exp453"]
        catalog_entry = next(item for item in catalog["experiments"] if item["id"] == "Exp453")
        champion_entry = next(item for item in champion["history"] if item["experiment"] == "Exp453")
        self.assertEqual(sum(item["id"] == "Exp453" for item in catalog["experiments"]), 1)
        self.assertEqual(sum(item["experiment"] == "Exp453" for item in champion["history"]), 1)
        for record in (exp453, catalog_entry, champion_entry):
            self.assertEqual(record["status"], "official_readiness_passed_evaluation_failed_immutable_no_retry")
            self.assertEqual(record["predecessor_prediction_sha256"], PREDICTION_SHA256_ANTIGRAVITY)
            self.assertEqual(record["target_dataset_count"], 48)
            self.assertEqual(record["target_prediction_count"], 96)
            self.assertEqual(record["target_family_method_pair_count"], 16)
            self.assertEqual(record["alpha"], 0.005)
            self.assertEqual(record["fixed_executable"], FIXED_EXECUTABLE_ANTIGRAVITY)
            self.assertEqual(record["fixed_prefix"], FIXED_PREFIX_ANTIGRAVITY)
            self.assertEqual(record["runtime_versions"], {"python": "3.14.6", "numpy": "2.3.5", "aeon": "1.4.0", "scikit_learn": "1.7.2"})
            self.assertEqual(record["label_access_count"], 0)
            self.assertTrue(record["official_readiness_started"])
            self.assertTrue(record["official_evaluate_started"])
            self.assertFalse(record["official_inspect_started"])
            self.assertEqual(record["official_readiness_invocations"], 1)
            self.assertEqual(record["official_readiness_exit_code"], 0)
            self.assertEqual(record["official_evaluate_invocations"], 1)
            self.assertEqual(record["official_evaluate_exit_code"], 1)
            self.assertEqual(record["official_inspect_invocations"], 0)
            self.assertEqual(record["label_sessions"], 0)
            self.assertEqual(record["label_queries"], 0)
            self.assertEqual(record["label_rows"], 0)
            self.assertIsNone(record["result_sha256"])
            self.assertIsNone(record["evaluation_sha256"])
            self.assertIsNone(record["decision_sha256"])
            self.assertIsNone(record["inspect_sha256"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
            self.assertEqual(record["official_failure"]["phase"], "evaluate")
            self.assertFalse(record["official_failure"]["evaluation_root_created"])
            self.assertFalse(record["official_failure"]["retry_permitted"])
        self.assertEqual(champion["pending_candidate"], state["pending_candidate"])
        self.assertFalse((ROOT_ANTIGRAVITY / "EXPERIMENT_HISTORY.json").exists())

    def test_document_hash_bindings_are_exact_antigravity(self) -> None:
        """Break caught: readiness later consumes document bytes different from registered bytes."""
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        assert isinstance(state, dict)
        bindings = state["exp453"]["document_sha256"]
        for key, relative in zip(("preregistration", "runbook", "implementation_report"), DOCUMENTS_ANTIGRAVITY, strict=True):
            with self.subTest(key=key):
                actual = hashlib.sha256((ROOT_ANTIGRAVITY / relative).read_bytes()).hexdigest()
                self.assertEqual(bindings[key], actual)

    def test_map_is_bounded_strict_and_records_task_five_cli_antigravity(self) -> None:
        """Break caught: Task 5 map hides the closed CLI or keeps a false future node."""
        builder = map_builder_antigravity()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict)
        self.assertLessEqual(len(graph["nodes"]), 20)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertIn("task5_cli", node_by_id)
        self.assertNotEqual(node_by_id["task5_cli"].get("status"), "future_unknown")
        self.assertTrue(node_by_id["task5_cli"]["evidence"])
        self.assertFalse(graph["unknown_relationships"])
        self.assertTrue(any(edge["from"] == "task5_cli" and edge["to"] == "task3_readiness" and edge["type"] == "reads" for edge in graph["edges"]))
        self.assertTrue(any(edge["from"] == "task5_cli" and edge["to"] == "task4_evaluation_replay" and edge["type"] == "reads" for edge in graph["edges"]))
        match = re.search(r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>', rendered, flags=re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        embedded = json.loads(html_module.unescape(match.group(1)))
        self.assertEqual(embedded, graph)
        builder.validate_map_antigravity(graph, lock, rendered)

    def test_map_answers_task_two_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: implemented foundations remain hidden behind the old future node."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        expected = {
            "task2_config": "test_config_antigravity.py",
            "task2_contracts": "test_contracts_antigravity.py",
            "task2_artifacts": "test_artifacts_antigravity.py",
            "task2_bootstrap": "test_bootstrap_antigravity.py",
            "task2_dashboard": "test_dashboard_export_antigravity.py",
        }
        for node_id, test_leaf in expected.items():
            with self.subTest(node_id=node_id):
                self.assertIn(node_id, node_by_id)
                self.assertNotEqual(node_by_id[node_id].get("status"), "future_unknown")
                self.assertIn("caller", node_by_id[node_id]["role"].lower())
                self.assertIn("effect", node_by_id[node_id]["role"].lower())
                self.assertTrue(any(path.endswith(test_leaf) for path in node_by_id[node_id]["tests"]))
                self.assertTrue(any(path.endswith("test_controlled_fork_differential_antigravity.py") for path in node_by_id[node_id]["tests"]))
                self.assertTrue(any(edge["from"] == "task2_regressions" and edge["to"] == node_id and edge["type"] == "tests" for edge in graph["edges"]))

    def test_map_answers_task_three_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: Task 3 runtime/predecessor/readiness remain hidden behind a future node."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        expected = {
            "task3_runtime_binding": "test_runtime_binding_antigravity.py",
            "task3_predecessor": "test_predecessor_antigravity.py",
            "task3_sources": "test_sources_antigravity.py",
            "task3_lifecycle": "test_lifecycle_antigravity.py",
            "task3_readiness": "test_readiness_antigravity.py",
        }
        for node_id, test_leaf in expected.items():
            with self.subTest(node_id=node_id):
                self.assertIn(node_id, node_by_id)
                self.assertNotEqual(node_by_id[node_id].get("status"), "future_unknown")
                self.assertIn("caller", node_by_id[node_id]["role"].lower())
                self.assertIn("effect", node_by_id[node_id]["role"].lower())
                self.assertTrue(any(path.endswith(test_leaf) for path in node_by_id[node_id]["tests"]))
                self.assertTrue(node_by_id[node_id]["evidence"])

    def test_map_answers_task_four_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: Task 4 evaluator/replay remain hidden behind a future node."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        node = node_by_id["task4_evaluation_replay"]
        self.assertNotEqual(node.get("status"), "future_unknown")
        self.assertIn("caller", node["role"].lower())
        self.assertIn("effect", node["role"].lower())
        self.assertTrue(any(path.endswith("test_evaluation_antigravity.py") for path in node["tests"]))
        self.assertTrue(any(path.endswith("test_replay_antigravity.py") for path in node["tests"]))
        self.assertTrue(node["evidence"])
        self.assertTrue(any(edge["from"] == "implementation_plan" and edge["to"] == "task4_evaluation_replay" and edge["type"] == "governs" for edge in graph["edges"]))
        self.assertTrue(any(edge["from"] == "task4_evaluation_replay" and edge["to"] == "task3_predecessor" and edge["type"] == "reads" for edge in graph["edges"]))
        self.assertTrue(any(edge["from"] == "task4_evaluation_replay" and edge["to"] == "task3_sources" and edge["type"] == "reads" for edge in graph["edges"]))

    def test_map_answers_task_five_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: Task 5 closed CLI remains hidden behind a future node."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        node = {item["id"]: item for item in graph["nodes"]}["task5_cli"]
        self.assertNotEqual(node.get("status"), "future_unknown")
        self.assertIn("caller", node["role"].lower())
        self.assertIn("effect", node["role"].lower())
        self.assertTrue(any(path.endswith("test_cli_antigravity.py") for path in node["tests"]))
        self.assertTrue(node["evidence"])
        self.assertTrue(any(edge["from"] == "implementation_plan" and edge["to"] == "task5_cli" and edge["type"] == "governs" for edge in graph["edges"]))
        self.assertTrue(any(edge["from"] == "task3_runtime_binding" and edge["to"] == "task5_cli" and edge["type"] == "reads" for edge in graph["edges"]))
        self.assertTrue((ROOT_ANTIGRAVITY / "experiments/exp453_exp452_fixed_interpreter_binding_repair_antigravity/cli_antigravity.py").is_file())


if __name__ == "__main__":
    unittest.main()
