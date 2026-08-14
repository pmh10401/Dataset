"""Governance contract for the planned, label-free Exp454 authority-boundary repair."""

from __future__ import annotations

import hashlib
import html as html_module
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PLAN_ANTIGRAVITY = "docs/superpowers/plans/2026-08-14-exp454-exp449-authority-boundary-repair-antigravity.html"
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP454_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP454_RUNBOOK_antigravity_KO.html",
    "docs/EXP454_IMPLEMENTATION_REPORT_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp454_antigravity.html",
    "docs/codemap/codemap_exp454_antigravity.json",
    "docs/codemap/codemap_exp454_antigravity.lock",
)
FIXED_EXECUTABLE_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
FIXED_PREFIX_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
PREDICTION_SHA256_ANTIGRAVITY = "a6054a44fc4e78f354dcf61d2e003284804680cd4197405364fcb5ef8d822e0d"
FAILURE_AUDIT_SHA256_ANTIGRAVITY = "97e7bea79bd1a830b925e7ffa2234e4e34725ac6734e080216f409688cb58276"
ARTIFACT_MAP_SHA256_ANTIGRAVITY = "748fdc62de555279640b4ce588b3155670ebe33e13e5997d40061d96011bff65"
EVALUATION_SHA256_ANTIGRAVITY = "c2b6321d18ba4c98e426be96ea3d3fb0824e75926fa54ba4600634dded68e1ac"
DECISION_SHA256_ANTIGRAVITY = "3510a426d3ee083ccb3f8599c64d0c91cc56d0b902780c8b6a2136098cd9a595"
INSPECT_SHA256_ANTIGRAVITY = "e76cf5d55af3472fb98c4f0773fe722d3a910667bb068a5df102767c44b9b6c0"
REVIEWED_COMMIT_ANTIGRAVITY = "32240c4e71776633249549a3d1d88b0e404d9463"
DESIGN_COMMIT_ANTIGRAVITY = "2a8486c42b385f4e589c7062a1c16f4be4372f4f"
DESIGN_SHA256_ANTIGRAVITY = "b76806b3ff9e89abfa4547f0ab5bb3ef707c28713b39675a2d1afad9e933f78f"
KIND_ANTIGRAVITY = "exp453_exp449_authority_boundary_repair"
RUN_IDENTITY_ANTIGRAVITY = "exp454-exp453-exp449-authority-boundary-repair-antigravity-20260814"
TRACKED_FINGERPRINT_PATHS_ANTIGRAVITY = (
    "CHAMPION_LEDGER.json",
    "EXPERIMENT_CATALOG.json",
    "FAILURE_LEDGER.json",
    "RESEARCH_STATE.json",
    "docs/EXP454_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP454_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP454_RUNBOOK_antigravity_KO.html",
    "docs/superpowers/plans/2026-08-14-exp454-exp449-authority-boundary-repair-antigravity.html",
    "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_governance_antigravity.py",
    "scratch/build_codemap_exp454_antigravity.py",
)


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


def map_builder_antigravity(relative: str, name: str) -> object:
    path = ROOT_ANTIGRAVITY / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{relative} is not importable")
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
            if task_number in (1, 2, 3, 4, 5):
                for tag in tags:
                    self.assertRegex(tag, r"\bchecked(?:\s|=|>)")
            else:
                self.assertEqual(
                    sum(bool(re.search(r"\bchecked(?:\s|=|>)", tag)) for tag in tags),
                    6,
                )

    def test_planned_truth_preserves_science_and_champion_antigravity(self) -> None:
        """Break caught: governance changes the frozen experiment or claims an official result."""
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        failures = strict_json_antigravity(ROOT_ANTIGRAVITY / "FAILURE_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict) and isinstance(failures, dict)
        exp454 = state["exp454"]
        catalog_entry = next(item for item in catalog["experiments"] if item["id"] == "Exp454")
        champion_entry = next(item for item in champion["history"] if item["experiment"] == "Exp454")
        self.assertEqual(sum(item["id"] == "Exp454" for item in catalog["experiments"]), 1)
        self.assertEqual(sum(item["experiment"] == "Exp454" for item in champion["history"]), 1)
        exp454_failure = next(item for item in failures["entries"] if item.get("experiment") == "Exp454")
        self.assertEqual(sum(item.get("experiment") == "Exp454" for item in failures["entries"]), 1)
        for record in (exp454, catalog_entry, champion_entry):
            self.assertEqual(record["status"], "official_evaluation_complete_review_reinspection_failed_stopped_immutable")
            self.assertEqual(record["kind"] if "kind" in record else KIND_ANTIGRAVITY, KIND_ANTIGRAVITY)
            self.assertEqual(record["run_identity"], RUN_IDENTITY_ANTIGRAVITY)
            self.assertEqual(record["predecessor_prediction_sha256"], PREDICTION_SHA256_ANTIGRAVITY)
            self.assertEqual(record["predecessor_failure_audit_sha256"], FAILURE_AUDIT_SHA256_ANTIGRAVITY)
            self.assertEqual(record["target_dataset_count"], 48)
            self.assertEqual(record["target_prediction_count"], 96)
            self.assertEqual(record["target_family_method_pair_count"], 16)
            self.assertEqual(record["seeds"], [20260721, 20261721])
            self.assertEqual(record["alpha"], 0.005)
            self.assertEqual(record["fixed_executable"], FIXED_EXECUTABLE_ANTIGRAVITY)
            self.assertEqual(record["fixed_prefix"], FIXED_PREFIX_ANTIGRAVITY)
            self.assertEqual(record["runtime_versions"], {"python": "3.14.6", "numpy": "2.3.5", "aeon": "1.4.0", "scikit_learn": "1.7.2"})
            self.assertEqual(record["official_reviewed_commit"], REVIEWED_COMMIT_ANTIGRAVITY)
            self.assertEqual(record["label_access_count"], 48)
            self.assertTrue(record["official_readiness_started"])
            self.assertTrue(record["official_evaluate_started"])
            self.assertTrue(record["official_inspect_started"])
            self.assertEqual(record["official_readiness_invocations"], 1)
            self.assertEqual(record["official_evaluate_invocations"], 1)
            self.assertEqual(record["official_inspect_invocations"], 2)
            self.assertEqual(record["official_readiness_exit_code"], 0)
            self.assertEqual(record["official_evaluate_exit_code"], 0)
            self.assertEqual(record["official_inspect_exit_codes"], [0, 1])
            self.assertEqual(record["official_inspect_successes"], 1)
            self.assertEqual(record["official_inspect_failures"], 1)
            self.assertEqual(record["label_sessions"], 1)
            self.assertEqual(record["label_queries"], 48)
            self.assertEqual(record["label_rows"], 72300)
            self.assertEqual(record["result_sha256"], ARTIFACT_MAP_SHA256_ANTIGRAVITY)
            self.assertEqual(record["artifact_map_sha256"], ARTIFACT_MAP_SHA256_ANTIGRAVITY)
            self.assertEqual(record["evaluation_sha256"], EVALUATION_SHA256_ANTIGRAVITY)
            self.assertEqual(record["decision_sha256"], DECISION_SHA256_ANTIGRAVITY)
            self.assertEqual(record["inspect_sha256"], INSPECT_SHA256_ANTIGRAVITY)
            self.assertEqual(record["qualified_count"], 2)
            self.assertEqual(record["qualified_candidates"], ["ECG5000:hydra_local_gap", "ChlorineConcentration:multirocket_local_gap"])
            self.assertTrue(record["routing_research_eligible"])
            self.assertTrue(record["permanent_stop"])
            self.assertEqual(record["stopped_receipt_sha256"], "b6bab58bd99c8857b5476d578c4f8cf4719569b6e58e99cd82bd80628ebb3d3d")
            self.assertEqual(record["failure_audit_sha256"], "3605405e08270bb7854e0e8d8a68e58760fd9f6f577634d37f029be0c8760b33")
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
            self.assertFalse(record["scientific_winner_selected"])
        self.assertEqual(exp454["kind"], KIND_ANTIGRAVITY)
        self.assertEqual(catalog_entry["kind"], KIND_ANTIGRAVITY)
        self.assertEqual(exp454["reviewed_design_commit"], DESIGN_COMMIT_ANTIGRAVITY)
        self.assertEqual(exp454["reviewed_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
        self.assertEqual(catalog_entry["reviewed_design_commit"], DESIGN_COMMIT_ANTIGRAVITY)
        self.assertEqual(catalog_entry["reviewed_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], state["pending_candidate"])
        self.assertFalse((ROOT_ANTIGRAVITY / "EXPERIMENT_HISTORY.json").exists())
        exp453_failure = next(item for item in failures["entries"] if item["experiment"] == "Exp453")
        self.assertEqual(exp453_failure["facts"]["failure_audit_sha256"], FAILURE_AUDIT_SHA256_ANTIGRAVITY)
        self.assertEqual(exp454_failure["facts"]["artifact_map_sha256"], ARTIFACT_MAP_SHA256_ANTIGRAVITY)
        self.assertEqual(exp454_failure["facts"]["inspect_exit_codes"], [0, 1])
        self.assertTrue(exp454_failure["facts"]["result_root_unchanged_after_failure"])

    def test_document_hash_bindings_are_exact_antigravity(self) -> None:
        """Break caught: readiness later consumes document bytes different from registered bytes."""
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)
        catalog_entry = next(item for item in catalog["experiments"] if item["id"] == "Exp454")
        champion_entry = next(item for item in champion["history"] if item["experiment"] == "Exp454")
        for bindings in (
            state["exp454"]["document_sha256"],
            catalog_entry["document_sha256"],
            champion_entry["document_sha256"],
        ):
            for key, relative in zip(("preregistration", "runbook", "implementation_report"), DOCUMENTS_ANTIGRAVITY, strict=True):
                with self.subTest(record=key, relative=relative):
                    actual = hashlib.sha256((ROOT_ANTIGRAVITY / relative).read_bytes()).hexdigest()
                    self.assertEqual(bindings[key], actual)

    def test_inherited_exp453_map_is_stale_invalid_source_parent_antigravity(self) -> None:
        """Break caught: Task 1 proceeds as if the inherited Exp453 map still binds HEAD."""
        builder = map_builder_antigravity("scratch/build_codemap_exp453_antigravity.py", "exp453_map_builder_antigravity")
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp453_antigravity.json")
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp453_antigravity.lock")
        rendered = (ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp453_antigravity.html").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid source parent"):
            builder.validate_map_antigravity(graph, lock, rendered)
        report = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]).read_text(encoding="utf-8")
        self.assertIn("invalid source parent", report)
        self.assertIn("stale", report.lower())
        self.assertIn("sources", report)
        self.assertIn("readiness", report)
        self.assertIn("evaluator", report)
        self.assertIn("dual authority", report)
        self.assertIn("predecessor", report)

    def test_map_is_bounded_strict_and_records_task_five_cli_antigravity(self) -> None:
        """Break caught: Task 5 map hides the closed CLI or keeps a false future node."""
        builder = map_builder_antigravity("scratch/build_codemap_exp454_antigravity.py", "exp454_map_builder_antigravity")
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict) and isinstance(lock, dict)
        self.assertLessEqual(len(graph["nodes"]), 20)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertNotEqual(node_by_id["task4_evaluation_replay"].get("status"), "future_unknown")
        self.assertTrue(node_by_id["task4_evaluation_replay"].get("evidence"))
        self.assertIn("task5_cli", node_by_id)
        self.assertNotEqual(node_by_id["task5_cli"].get("status"), "future_unknown")
        self.assertTrue(node_by_id["task5_cli"]["evidence"])
        self.assertNotIn("task3_runtime_predecessors", node_by_id)
        self.assertFalse(graph["unknown_relationships"])
        self.assertIn("tracked_fingerprints", lock)
        self.assertEqual(len(lock["tracked_fingerprints"]), 10)
        self.assertEqual(set(lock["tracked_fingerprints"]), set(TRACKED_FINGERPRINT_PATHS_ANTIGRAVITY))
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
            "task2_predecessor": "test_predecessor_antigravity.py",
        }
        for node_id, test_leaf in expected.items():
            with self.subTest(node_id=node_id):
                self.assertIn(node_id, node_by_id)
                self.assertNotEqual(node_by_id[node_id].get("status"), "future_unknown")
                self.assertIn("caller", node_by_id[node_id]["role"].lower())
                self.assertIn("effect", node_by_id[node_id]["role"].lower())
                self.assertTrue(any(path.endswith(test_leaf) for path in node_by_id[node_id]["tests"]))
                if node_id != "task2_predecessor":
                    self.assertTrue(any(path.endswith("test_controlled_fork_differential_antigravity.py") for path in node_by_id[node_id]["tests"]))
                    self.assertTrue(any(edge["from"] == "task2_regressions" and edge["to"] == node_id and edge["type"] == "tests" for edge in graph["edges"]))
        predecessor = node_by_id["task2_predecessor"]
        self.assertIn("normalize_exp449_target_authority_antigravity", predecessor["role"])
        self.assertIn("sources", predecessor["role"].lower())
        self.assertIn("readiness", predecessor["role"].lower())
        self.assertIn("evaluator", predecessor["role"].lower())
        self.assertFalse(any(path.endswith("test_controlled_fork_differential_antigravity.py") for path in predecessor["tests"]))
        self.assertIn("task2_predecessor_tests", node_by_id)
        self.assertEqual(node_by_id["task2_predecessor_tests"]["path"], "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/tests/test_predecessor_antigravity.py")
        self.assertTrue(
            any(
                edge["from"] == "task2_predecessor_tests"
                and edge["to"] == "task2_predecessor"
                and edge["type"] == "tests"
                and edge["evidence"]["path"].endswith("test_predecessor_antigravity.py")
                for edge in graph["edges"]
            )
        )
        self.assertFalse(any(edge["from"] == "task2_regressions" and edge["to"] == "task2_predecessor" for edge in graph["edges"]))

    def test_map_answers_task_three_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: implemented runtime/lifecycle/readiness remain hidden behind the old future node."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        expected = {
            "task3_runtime_binding": "test_runtime_binding_antigravity.py",
            "task3_sources": "test_sources_antigravity.py",
            "task3_lifecycle_readiness": "test_readiness_antigravity.py",
        }
        for node_id, test_leaf in expected.items():
            with self.subTest(node_id=node_id):
                self.assertIn(node_id, node_by_id)
                self.assertNotEqual(node_by_id[node_id].get("status"), "future_unknown")
                self.assertIn("caller", node_by_id[node_id]["role"].lower())
                self.assertIn("effect", node_by_id[node_id]["role"].lower())
                self.assertTrue(any(path.endswith(test_leaf) for path in node_by_id[node_id]["tests"]))
        self.assertTrue(any(path.endswith("test_lifecycle_antigravity.py") for path in node_by_id["task3_lifecycle_readiness"]["tests"]))
        self.assertTrue(
            any(
                edge["from"] == "task3_sources"
                and edge["to"] == "task2_predecessor"
                and edge["type"] == "reads"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task3_lifecycle_readiness"
                and edge["to"] == "task2_predecessor"
                and edge["type"] == "reads"
                for edge in graph["edges"]
            )
        )
        predecessor = node_by_id["task2_predecessor"]
        self.assertIn("sources", predecessor["role"].lower())
        self.assertIn("readiness", predecessor["role"].lower())
        self.assertTrue(any(path.endswith("test_sources_antigravity.py") for path in predecessor["tests"]))

    def test_map_answers_task_four_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: implemented evaluator/replay remain hidden behind the old future node."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        node = node_by_id["task4_evaluation_replay"]
        self.assertNotEqual(node.get("status"), "future_unknown")
        self.assertIn("caller", node["role"].lower())
        self.assertIn("effect", node["role"].lower())
        self.assertTrue(any(path.endswith("test_evaluation_antigravity.py") for path in node["tests"]))
        self.assertTrue(any(path.endswith("test_replay_antigravity.py") for path in node["tests"]))
        self.assertTrue(node["path"].endswith("evaluation_antigravity.py"))
        self.assertIn("capture_normalized_target_authority_antigravity", node["role"])
        self.assertTrue(
            any(
                edge["from"] == "task4_evaluation_replay"
                and edge["to"] == "task2_predecessor"
                and edge["type"] == "reads"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_evaluation_replay"
                and edge["to"] == "task3_sources"
                and edge["type"] == "reads"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "implementation_plan"
                and edge["to"] == "task4_evaluation_replay"
                and edge["type"] == "governs"
                for edge in graph["edges"]
            )
        )
        predecessor = node_by_id["task2_predecessor"]
        self.assertIn("evaluator", predecessor["role"].lower())
        self.assertTrue(any(path.endswith("test_evaluation_antigravity.py") for path in predecessor["tests"]))

    def test_map_answers_task_five_callers_effects_and_tests_antigravity(self) -> None:
        """Break caught: Task 5 closed CLI remains hidden behind a future node."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        node = {item["id"]: item for item in graph["nodes"]}["task5_cli"]
        self.assertNotEqual(node.get("status"), "future_unknown")
        self.assertIn("caller", node["role"].lower())
        self.assertIn("effect", node["role"].lower())
        self.assertTrue(any(path.endswith("test_cli_antigravity.py") for path in node["tests"]))
        self.assertTrue(node["path"].endswith("cli_antigravity.py"))
        self.assertIn("capture_normalized_target_authority_antigravity", node["role"])
        self.assertTrue(node["evidence"])
        self.assertTrue(
            any(
                edge["from"] == "implementation_plan"
                and edge["to"] == "task5_cli"
                and edge["type"] == "governs"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task5_cli"
                and edge["to"] == "task3_lifecycle_readiness"
                and edge["type"] == "reads"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task5_cli"
                and edge["to"] == "task4_evaluation_replay"
                and edge["type"] == "reads"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task3_runtime_binding"
                and edge["to"] == "task5_cli"
                and edge["type"] == "reads"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            (
                ROOT_ANTIGRAVITY
                / "experiments/exp454_exp453_exp449_authority_boundary_repair_antigravity/cli_antigravity.py"
            ).is_file()
        )


if __name__ == "__main__":
    unittest.main()
