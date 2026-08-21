from __future__ import annotations

import hashlib
import html.parser
import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from typing import Any


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PACKAGE_ANTIGRAVITY = (
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity"
)
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/"
    "2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html"
)
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/"
    "2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity-design.html"
)
TASK4_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/"
    "task-4-report-antigravity.md"
)
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP464_DASHBOARD_antigravity.html",
    "docs/EXP464_PREREGISTRATION_antigravity.html",
    "docs/EXP464_RUNBOOK_antigravity.html",
    "docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html",
)
FOUNDATION_FILES_ANTIGRAVITY = (
    f"{PACKAGE_ANTIGRAVITY}/config_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/centered_ramp_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/control_row_evidence_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/contracts_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/predecessor_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/qualification_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/raw_evidence_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/evaluation_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/artifacts_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/replay_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/dashboard_export_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/scoring_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/sources_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/exp463_control_row_evidence_antigravity.jsonl",
)
TEST_FILES_ANTIGRAVITY = (
    f"{PACKAGE_ANTIGRAVITY}/tests/test_config_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_contracts_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_predecessor_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_sources_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_centered_ramp_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_scoring_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_qualification_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_evaluation_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_artifacts_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_replay_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_dashboard_export_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_governance_antigravity.py",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp464_antigravity.html",
    "docs/codemap/codemap_exp464_antigravity.json",
    "docs/codemap/codemap_exp464_antigravity.lock",
)
RUN_ROOT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/"
    "exp464_ecg_residual_centered_ramp_confirmation_antigravity_20260821"
)
DESIGN_SHA256_ANTIGRAVITY = (
    "d0e36ae176598810289fbd5693a5e582ca6f0821a8b8fcd52edd1bcb96ec1f9f"
)
PRIMARY_CANDIDATE_ANTIGRAVITY = "template_residual_centered_ramp_confirmed_max"
STATUS_ANTIGRAVITY = "task_4_implemented_awaiting_fresh_review"
METHOD_ORDER_ANTIGRAVITY = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
EXPECTED_CHECKBOX_COUNT_ANTIGRAVITY = 42
EXPECTED_CHECKED_COUNT_ANTIGRAVITY = 21
EXPECTED_NEXT_ACTION_ANTIGRAVITY = "fresh_task_4_review_before_task_5"
EXPECTED_PENDING_ANTIGRAVITY = {
    "evaluation": None,
    "experiment": None,
    "prediction_sha256": None,
    "status": "none_exp446_publication_inspected_complete_not_pending_champion",
}
EXPECTED_NODE_IDS_ANTIGRAVITY = {
    "approved_design",
    "implementation_plan",
    "dashboard",
    "preregistration",
    "runbook",
    "implementation_report",
    "research_state",
    "experiment_catalog",
    "champion_ledger",
    "governance_test",
    "map_builder",
    "task2_config",
    "task2_control_row_evidence_file",
    "task2_control_row_evidence_module",
    "task2_contracts",
    "task2_predecessor",
    "task2_sources",
    "task3_raw_evidence",
    "task3_centered_ramp_feature",
    "task3_scoring",
    "task4_qualification",
    "task4_evaluation",
    "task4_artifacts",
    "task4_replay",
    "task4_dashboard_export",
    "task4_qualification_test",
    "task4_evaluation_test",
    "task4_artifacts_test",
    "task4_replay_test",
    "task4_dashboard_test",
    "future_task5_private_qualification",
}
EXPECTED_TRACKED_FINGERPRINTS_ANTIGRAVITY = {
    f"{PACKAGE_ANTIGRAVITY}/control_row_evidence_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/raw_evidence_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/exp463_control_row_evidence_antigravity.jsonl",
}
EXPECTED_EDGE_TRIPLES_ANTIGRAVITY = {
    ("task2_control_row_evidence_file", "freezes_inputs_for", "task2_control_row_evidence_module"),
    ("task2_control_row_evidence_module", "feeds", "task2_contracts"),
    ("task2_control_row_evidence_module", "feeds", "task2_predecessor"),
    ("task3_raw_evidence", "feeds", "task3_scoring"),
    ("task3_raw_evidence", "feeds", "task4_qualification"),
}
EXPECTED_FLOWS_ANTIGRAVITY = {
    "tracked Exp463 control evidence file -> control evidence loader -> contracts/predecessor",
    "raw evidence codecs -> Task 3 scoring -> Task 4 qualification/replay",
}
EXPECTED_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "method_fold_metrics_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


def strict_json_antigravity(path: Path) -> Any:
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


def load_builder_module_antigravity() -> object:
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp464_antigravity.py"
    if not path.is_file():
        raise AssertionError(f"missing map builder: {path}")
    spec = importlib.util.spec_from_file_location(
        "exp464_map_builder_antigravity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp464 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class CheckboxParserAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.checkbox_count = 0
        self.checked_count = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "input":
            return
        attr_map = dict(attrs)
        if attr_map.get("type") != "checkbox":
            return
        self.checkbox_count += 1
        if "checked" in attr_map:
            self.checked_count += 1


class GovernanceAntigravityTest(unittest.TestCase):
    maxDiff = None

    def _records_antigravity(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        self.assertIsInstance(state, dict)
        self.assertIsInstance(catalog, dict)
        self.assertIsInstance(champion, dict)
        assert isinstance(state, dict)
        assert isinstance(catalog, dict)
        assert isinstance(champion, dict)
        self.assertIn("exp464", state)
        catalog_matches = [
            row for row in catalog["experiments"] if row.get("id") == "Exp464"
        ]
        champion_matches = [
            row for row in champion["history"] if row.get("experiment") == "Exp464"
        ]
        self.assertEqual(len(catalog_matches), 1)
        self.assertEqual(len(champion_matches), 1)
        return state["exp464"], catalog_matches[0], champion_matches[0]

    def test_exp464_task4_truth_is_review_pending_without_results_antigravity(self) -> None:
        """Break caught: Task 4 code lands but repo truth, map, or boundaries drift."""
        required_files = (
            (PLAN_ANTIGRAVITY, DESIGN_ANTIGRAVITY, TASK4_REPORT_ANTIGRAVITY)
            + DOCUMENTS_ANTIGRAVITY
            + FOUNDATION_FILES_ANTIGRAVITY
            + TEST_FILES_ANTIGRAVITY
            + TRIPLET_ANTIGRAVITY
        )
        missing = [
            relative
            for relative in required_files
            if not (ROOT_ANTIGRAVITY / relative).is_file()
        ]
        self.assertEqual(missing, [], f"missing Task 4 artifacts: {missing}")
        for relative in required_files:
            path = ROOT_ANTIGRAVITY / relative
            self.assertIn("antigravity", path.name)

        self.assertEqual(
            hashlib.sha256(
                (ROOT_ANTIGRAVITY / DESIGN_ANTIGRAVITY).read_bytes(),
            ).hexdigest(),
            DESIGN_SHA256_ANTIGRAVITY,
        )

        run_root = Path(RUN_ROOT_ANTIGRAVITY)
        self.assertFalse(run_root.exists(), RUN_ROOT_ANTIGRAVITY)
        self.assertFalse(run_root.is_symlink(), RUN_ROOT_ANTIGRAVITY)

        state_record, catalog_record, champion_record = self._records_antigravity()
        for record in (state_record, catalog_record, champion_record):
            self.assertEqual(record["experiment"], "Exp464")
            self.assertEqual(
                record["kind"],
                "ecg_residual_centered_ramp_confirmation_qualification",
            )
            self.assertEqual(record["status"], STATUS_ANTIGRAVITY)
            self.assertEqual(record["implementation_status"], STATUS_ANTIGRAVITY)
            self.assertEqual(record["run_root"], RUN_ROOT_ANTIGRAVITY)
            self.assertEqual(record["approved_design"], DESIGN_ANTIGRAVITY)
            self.assertEqual(record["approved_plan"], PLAN_ANTIGRAVITY)
            self.assertEqual(
                record["approved_design_sha256"],
                DESIGN_SHA256_ANTIGRAVITY,
            )
            self.assertEqual(
                record["scientific_contract"]["method_order"],
                list(METHOD_ORDER_ANTIGRAVITY),
            )
            self.assertEqual(
                record["scientific_contract"]["primary_candidate"],
                PRIMARY_CANDIDATE_ANTIGRAVITY,
            )
            self.assertEqual(record["official_evaluate_count"], 0)
            self.assertEqual(record["official_inspect_count"], 0)
            self.assertEqual(record["private_run_count"], 0)
            self.assertFalse(record["result_root_created"])
            self.assertIsNone(record["official_result"])
            self.assertIsNone(record["prediction_sha256"])
            self.assertIsNone(record["evaluation_sha256"])
            self.assertIsNone(record["decision_sha256"])
            self.assertIsNone(record["artifact_map_sha256"])
            self.assertIsNone(record["dashboard_snapshot_sha256"])
            self.assertFalse(record["test_labels_opened"])
            self.assertEqual(record["label_query_count"], 0)
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
            self.assertFalse(record["scientific_winner_selected"])
            self.assertFalse(record["official_root_present"])
            self.assertFalse(record["review_only"])

        self.assertEqual(state_record["next_action"], EXPECTED_NEXT_ACTION_ANTIGRAVITY)
        self.assertEqual(catalog_record["notes"].count("21/42"), 1)
        self.assertIn("Task 4 result pipeline", state_record["notes"])
        self.assertIn("fresh Task 4 review", state_record["notes"])
        self.assertIn("result pipeline", champion_record["reason"])

        research_state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        self.assertEqual(research_state["active_experiment"], "Exp464")
        self.assertEqual(research_state["active_phase"], STATUS_ANTIGRAVITY)
        self.assertEqual(research_state["status"], STATUS_ANTIGRAVITY)
        self.assertEqual(research_state["pending_candidate"], EXPECTED_PENDING_ANTIGRAVITY)

        plan_text = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        self.assertIn(STATUS_ANTIGRAVITY, plan_text)
        self.assertIn("21/42", plan_text)
        self.assertIn("focused suite 16 tests", plan_text)
        checkbox_parser = CheckboxParserAntigravity()
        checkbox_parser.feed(plan_text)
        self.assertEqual(
            checkbox_parser.checkbox_count,
            EXPECTED_CHECKBOX_COUNT_ANTIGRAVITY,
        )
        self.assertEqual(
            checkbox_parser.checked_count,
            EXPECTED_CHECKED_COUNT_ANTIGRAVITY,
        )
        self.assertIn(
            '<h2><label><input type="checkbox" disabled> Task 4 — qualification, evaluation, artifacts, replay, dashboard export</label></h2>',
            plan_text,
        )

        parser = HtmlParserAntigravity()
        for relative in DOCUMENTS_ANTIGRAVITY + (PLAN_ANTIGRAVITY,):
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))

        report_text = (ROOT_ANTIGRAVITY / TASK4_REPORT_ANTIGRAVITY).read_text(
            encoding="utf-8",
        )
        for leaf in EXPECTED_LEAVES_ANTIGRAVITY:
            self.assertIn(leaf, report_text)

        builder = load_builder_module_antigravity()
        triplet_summary = builder.build_triplet_antigravity(validate_only=True)
        self.assertEqual(
            tuple(triplet_summary["triplet_paths"]),
            TRIPLET_ANTIGRAVITY,
        )
        self.assertIn(
            triplet_summary["source_parent"],
            triplet_summary["allowed_source_commits"],
        )

        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        html_text = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(
            encoding="utf-8",
        )
        self.assertEqual(graph["graph_schema"], "exp464_task4_results_codemap_v1")
        self.assertEqual(lock["lock_schema"], "exp464_task4_results_codemap_lock_v1")
        self.assertEqual(tuple(graph["triplet_paths"]), TRIPLET_ANTIGRAVITY)
        self.assertEqual(lock["generated_files"], list(TRIPLET_ANTIGRAVITY))
        self.assertEqual(graph["source_parent"], lock["source_parent"])
        self.assertEqual(graph["source_parent"], triplet_summary["source_parent"])
        node_ids = {node["id"] for node in graph["nodes"]}
        self.assertTrue(EXPECTED_NODE_IDS_ANTIGRAVITY.issubset(node_ids))
        self.assertEqual(graph["node_count"], len(graph["nodes"]))
        self.assertEqual(graph["edge_count"], len(graph["edges"]))
        self.assertEqual(lock["fingerprint_count"], len(lock["tracked_fingerprints"]))
        fingerprint_paths = {
            fingerprint["path"] for fingerprint in lock["tracked_fingerprints"]
        }
        self.assertTrue(
            EXPECTED_TRACKED_FINGERPRINTS_ANTIGRAVITY.issubset(fingerprint_paths)
        )
        edge_triples = {
            (edge["from"], edge["relation"], edge["to"]) for edge in graph["edges"]
        }
        self.assertTrue(EXPECTED_EDGE_TRIPLES_ANTIGRAVITY.issubset(edge_triples))
        self.assertTrue(EXPECTED_FLOWS_ANTIGRAVITY.issubset(set(graph["flows"])))
        self.assertIn("task_4_implemented_awaiting_fresh_review", html_text)
        self.assertIn("Task 5 private qualification", html_text)

        process_output = subprocess.check_output(
            ["ps", "-Ao", "pid=,command="],
            text=True,
        )
        active_processes = [
            line
            for line in process_output.splitlines()
            if "exp464_ecg_residual_centered_ramp_confirmation_antigravity" in line
            and "test_governance_antigravity" not in line
            and "build_codemap_exp464_antigravity.py" not in line
            and " -m unittest " not in line
            and " -m py_compile " not in line
        ]
        self.assertEqual(active_processes, [])


if __name__ == "__main__":
    unittest.main()
