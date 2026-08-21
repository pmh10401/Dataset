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
TASK5_REPORT_ANTIGRAVITY = (
    ".superpowers/sdd/"
    "2026-08-21-exp464-ecg-residual-centered-ramp-confirmation-antigravity.html/"
    "task-5-report-antigravity.md"
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
    f"{PACKAGE_ANTIGRAVITY}/cli_antigravity.py",
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
    f"{PACKAGE_ANTIGRAVITY}/tests/test_cli_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/tests/test_governance_antigravity.py",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp464_antigravity.html",
    "docs/codemap/codemap_exp464_antigravity.json",
    "docs/codemap/codemap_exp464_antigravity.lock",
)
OFFICIAL_ROOT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/"
    "exp464_ecg_residual_centered_ramp_confirmation_antigravity_20260821"
)
PRIVATE_ROOTS_ANTIGRAVITY = (
    "/private/tmp/exp464_task5_private_run1_antigravity",
    "/private/tmp/exp464_task5_private_run2_antigravity",
)
DESIGN_SHA256_ANTIGRAVITY = (
    "d0e36ae176598810289fbd5693a5e582ca6f0821a8b8fcd52edd1bcb96ec1f9f"
)
REVIEWED_COMMIT_ANTIGRAVITY = "b8646d0fa9bd57dcb1f6ab5e251064d34911a41c"
PREDECESSOR_COMMIT_ANTIGRAVITY = "147225831e9c13e9df5043fa88757bb03e01b79a"
PREDECESSOR_ARTIFACT_MAP_SHA256_ANTIGRAVITY = (
    "25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5"
)
PRIMARY_CANDIDATE_ANTIGRAVITY = "template_residual_centered_ramp_confirmed_max"
STATUS_ANTIGRAVITY = "stopped_before_official_evaluation_qualification_failed"
IMPLEMENTATION_STATUS_ANTIGRAVITY = (
    "task_5_private_train_reproducibility_complete_qualification_failed"
)
DECISION_STATUS_ANTIGRAVITY = "qualification_failed_blocked_before_routing_research"
METHOD_ORDER_ANTIGRAVITY = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
EXPECTED_CHECKBOX_COUNT_ANTIGRAVITY = 42
EXPECTED_CHECKED_COUNT_ANTIGRAVITY = 28
EXPECTED_NEXT_ACTION_ANTIGRAVITY = (
    "fresh_review_exp464_task5_private_qualification_failed_stop_no_task6"
)
EXPECTED_EXACT_NEXT_COMMAND_ANTIGRAVITY = (
    "fresh_sol_high_read_only_review_current_clean_head"
)
EXPECTED_INTERRUPTED_COMMAND_ANTIGRAVITY = (
    "none_exp464_task5_private_runs_complete_wait_for_fresh_review"
)
EXPECTED_OFFICIAL_COMMAND_ANTIGRAVITY = (
    "none_exp464_task6_forbidden_after_task5_qualification_failed"
)
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
    "task5_closed_cli",
    "task5_launcher",
    "task5_cli_test",
}
EXPECTED_TRACKED_FINGERPRINTS_ANTIGRAVITY = {
    f"{PACKAGE_ANTIGRAVITY}/control_row_evidence_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/raw_evidence_antigravity.py",
    f"{PACKAGE_ANTIGRAVITY}/exp463_control_row_evidence_antigravity.jsonl",
    f"{PACKAGE_ANTIGRAVITY}/cli_antigravity.py",
    TASK5_REPORT_ANTIGRAVITY,
}
EXPECTED_EDGE_TRIPLES_ANTIGRAVITY = {
    (
        "task2_control_row_evidence_file",
        "freezes_inputs_for",
        "task2_control_row_evidence_module",
    ),
    ("task2_control_row_evidence_module", "feeds", "task2_contracts"),
    ("task2_control_row_evidence_module", "feeds", "task2_predecessor"),
    ("task3_raw_evidence", "feeds", "task3_scoring"),
    ("task3_raw_evidence", "feeds", "task4_qualification"),
}
EXPECTED_FLOWS_ANTIGRAVITY = {
    "tracked Exp463 control evidence file -> control evidence loader -> contracts/predecessor",
    "raw evidence codecs -> Task 3 scoring -> Task 4 qualification/replay",
    "Task 5 launcher/test -> closed CLI -> Task 4 qualification/evaluation/artifacts/replay -> private non-official result truth",
    "Task 5 closed CLI -> two completed exact-equal private TRAIN reproductions -> fresh review gate",
}
EXPECTED_LEAVES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
)
EXPECTED_LEAF_SHA256_ANTIGRAVITY = {
    "artifact_sha256_antigravity.json": (
        "8e0cc3391f5acf970fb73a322a2ec313538cd2c4d153114e07b2d480f845bbca"
    ),
    "context_antigravity.json": (
        "577a8a989abaf7b64d69968578e3d15b2b597acb97618c7d71eeaaf45fc46653"
    ),
    "dashboard_snapshot_antigravity.json": (
        "da1e2333728b2886cea918f4c34ad6d084d10e1811b48c8903c0946ad167e1cb"
    ),
    "decision_antigravity.json": (
        "d66a23ef0749833220700f7d6b3ec56422722a9167d443cf2f58d327ce9593cd"
    ),
    "failed_folds_antigravity.jsonl": (
        "54eefdbb8bbc3d2495bfa1cd0d444c04ffa9845c5f12f4e79f6bc6e25717bff3"
    ),
    "method_fold_metrics_antigravity.jsonl": (
        "51073f3caf4793a2a70602547d6f249e42bcc7dcd64709ab5a62ccfeba791bcf"
    ),
    "method_summary_antigravity.jsonl": (
        "3f7ad67d51c7dd63a31e9f373b14eb2ce2ffc616a81a611db2dd599e083f5540"
    ),
}
EXPECTED_LEAF_BYTES_ANTIGRAVITY = {
    "artifact_sha256_antigravity.json": 605,
    "context_antigravity.json": 1180,
    "dashboard_snapshot_antigravity.json": 3839,
    "decision_antigravity.json": 428,
    "failed_folds_antigravity.jsonl": 5511663,
    "method_fold_metrics_antigravity.jsonl": 53043215,
    "method_summary_antigravity.jsonl": 3899,
}
EXPECTED_METHOD_PASS_COUNTS_ANTIGRAVITY = {
    "C0": {"passed_count": 9, "failed_count": 1},
    "C1": {"passed_count": 8, "failed_count": 2},
    "C2": {"passed_count": 9, "failed_count": 1},
    "C3": {"passed_count": 0, "failed_count": 10},
    "C4": {"passed_count": 3, "failed_count": 7},
    "C5": {"passed_count": 9, "failed_count": 1},
    "C6": {"passed_count": 9, "failed_count": 1},
    "C7": {"passed_count": 3, "failed_count": 7},
    "C8": {"passed_count": 10, "failed_count": 0},
}
EXPECTED_STDOUT_SHA256_ANTIGRAVITY = (
    "f41151e67a07b33bae607437501317e61c79d8f5508b62d0886b210b63180111"
)
EXPECTED_FAILURES_ANTIGRAVITY = (
    {
        "name": "reviewed_commit_mismatch",
        "stdout_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "stderr_sha256": (
            "8a3a459348603f9c79fd68fdd39cd39594dd87e63547e539a147bad29a4657cf"
        ),
        "time_sha256": (
            "e4a4cb4a2aa2fb6d1f64f82e137be621de64c2b92de18d41d25572d8949b90db"
        ),
        "root_created": False,
    },
    {
        "name": "duplicate_process_false_positive",
        "stdout_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "stderr_sha256": (
            "e67bee7ce606726713bd7aaa6df6a2cf978094c3d68123372b4f3d9bb23be1a4"
        ),
        "time_sha256": (
            "539f1596e3b178a02613d497aadf76b6bf99d0df3bab746117cc74e40e28b68a"
        ),
        "root_created": False,
    },
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
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        self.assertIsInstance(state, dict)
        self.assertIsInstance(catalog, dict)
        self.assertIsInstance(champion, dict)
        assert isinstance(state, dict)
        assert isinstance(catalog, dict)
        assert isinstance(champion, dict)
        catalog_matches = [
            row for row in catalog["experiments"] if row.get("id") == "Exp464"
        ]
        champion_matches = [
            row for row in champion["history"] if row.get("experiment") == "Exp464"
        ]
        self.assertEqual(len(catalog_matches), 1)
        self.assertEqual(len(champion_matches), 1)
        return state, state["exp464"], catalog_matches[0], champion_matches[0]

    def _assert_task5_result_antigravity(self, result: dict[str, Any]) -> None:
        self.assertEqual(result["reviewed_commit"], REVIEWED_COMMIT_ANTIGRAVITY)
        self.assertEqual(result["status"], DECISION_STATUS_ANTIGRAVITY)
        self.assertFalse(result["temporary_roots_removed"])
        self.assertEqual(result["row_count"], 90)
        self.assertEqual(result["failed_row_count"], 30)
        self.assertEqual(result["summary_count"], 9)
        self.assertEqual(result["leaf_count"], 7)
        self.assertEqual(result["stdout_sha256"], EXPECTED_STDOUT_SHA256_ANTIGRAVITY)
        self.assertEqual(
            result["artifact_map_sha256"],
            EXPECTED_LEAF_SHA256_ANTIGRAVITY["artifact_sha256_antigravity.json"],
        )
        self.assertEqual(result["primary_candidate"], PRIMARY_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(result["primary_passed_count"], 9)
        self.assertEqual(result["primary_required_count"], 10)
        self.assertEqual(result["label_query_count"], 0)
        self.assertFalse(result["official_root_exists_after"])
        self.assertEqual(result["private_roots"], list(PRIVATE_ROOTS_ANTIGRAVITY))
        self.assertEqual(result["leaf_sha256"], EXPECTED_LEAF_SHA256_ANTIGRAVITY)
        self.assertEqual(result["leaf_bytes"], EXPECTED_LEAF_BYTES_ANTIGRAVITY)
        self.assertEqual(
            result["method_pass_counts"],
            EXPECTED_METHOD_PASS_COUNTS_ANTIGRAVITY,
        )
        self.assertEqual(result["pre_science_failures_retained"], list(EXPECTED_FAILURES_ANTIGRAVITY))
        self.assertEqual(result["overlap_nonresult"], {"pre_science_failures_retained": 2, "writes": False})
        self.assertEqual(
            result["exact_equality"],
            {
                "artifact_map_hash": True,
                "dashboard": True,
                "decision": True,
                "failed_rows": True,
                "leaf_bytes": True,
                "leaf_hashes": True,
                "rows": True,
                "stdout_payload": True,
                "summaries": True,
            },
        )
        self.assertEqual(
            result["runs"],
            [
                {
                    "run": 1,
                    "evaluate_exit": 0,
                    "inspect_exit": 0,
                    "evaluate_seconds": 269.67,
                    "inspect_seconds": 6.62,
                },
                {
                    "run": 2,
                    "evaluate_exit": 0,
                    "inspect_exit": 0,
                    "evaluate_seconds": 261.5,
                    "inspect_seconds": 6.64,
                },
            ],
        )
        decision = result["decision"]
        self.assertEqual(decision["status"], DECISION_STATUS_ANTIGRAVITY)
        self.assertEqual(decision["primary_candidate"], PRIMARY_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(decision["primary_passed_count"], 9)
        self.assertEqual(decision["primary_required_count"], 10)
        self.assertFalse(decision["routing_applied"])
        self.assertFalse(decision["eligible_for_routing_research"])
        self.assertEqual(decision["label_query_count"], 0)
        self.assertFalse(decision["test_labels_opened"])
        self.assertEqual(
            decision["failure_reasons"],
            [
                "primary_candidate_not_10_of_10",
                "primary_gradual_drift_auc_below_min",
                "primary_gradual_drift_min_below_0.85",
            ],
        )

    def test_exp464_task5_private_runs_complete_and_task6_blocked_antigravity(
        self,
    ) -> None:
        """Break caught: private-run result truth, retained roots, or blocked Task 6 drift."""

        required_files = (
            (PLAN_ANTIGRAVITY, DESIGN_ANTIGRAVITY, TASK4_REPORT_ANTIGRAVITY, TASK5_REPORT_ANTIGRAVITY)
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
        self.assertEqual(missing, [], f"missing Exp464 Task 5 artifacts: {missing}")
        for relative in required_files:
            self.assertIn("antigravity", Path(relative).name)

        self.assertEqual(
            hashlib.sha256((ROOT_ANTIGRAVITY / DESIGN_ANTIGRAVITY).read_bytes()).hexdigest(),
            DESIGN_SHA256_ANTIGRAVITY,
        )

        official_root = Path(OFFICIAL_ROOT_ANTIGRAVITY)
        self.assertFalse(official_root.exists(), OFFICIAL_ROOT_ANTIGRAVITY)
        self.assertFalse(official_root.is_symlink(), OFFICIAL_ROOT_ANTIGRAVITY)
        for private_root_text in PRIVATE_ROOTS_ANTIGRAVITY:
            private_root = Path(private_root_text)
            self.assertTrue(private_root.is_dir(), private_root_text)
            self.assertFalse(private_root.is_symlink(), private_root_text)
            leaves = sorted(path.name for path in private_root.iterdir() if path.is_file())
            self.assertEqual(leaves, sorted(EXPECTED_LEAVES_ANTIGRAVITY))
            for leaf_name in EXPECTED_LEAVES_ANTIGRAVITY:
                leaf_path = private_root / leaf_name
                self.assertEqual(leaf_path.stat().st_size, EXPECTED_LEAF_BYTES_ANTIGRAVITY[leaf_name])
                self.assertEqual(
                    hashlib.sha256(leaf_path.read_bytes()).hexdigest(),
                    EXPECTED_LEAF_SHA256_ANTIGRAVITY[leaf_name],
                )

        research_state, state_record, catalog_record, champion_record = self._records_antigravity()
        for record in (state_record, catalog_record, champion_record):
            self.assertEqual(record["experiment"], "Exp464")
            self.assertEqual(
                record["kind"],
                "ecg_residual_centered_ramp_confirmation_qualification",
            )
            self.assertEqual(record["status"], STATUS_ANTIGRAVITY)
            self.assertEqual(
                record["implementation_status"],
                IMPLEMENTATION_STATUS_ANTIGRAVITY,
            )
            self.assertEqual(record["run_root"], OFFICIAL_ROOT_ANTIGRAVITY)
            self.assertEqual(record["approved_design"], DESIGN_ANTIGRAVITY)
            self.assertEqual(record["approved_plan"], PLAN_ANTIGRAVITY)
            self.assertEqual(record["approved_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
            self.assertEqual(
                record["scientific_contract"]["method_order"],
                list(METHOD_ORDER_ANTIGRAVITY),
            )
            self.assertEqual(
                record["scientific_contract"]["primary_candidate"],
                PRIMARY_CANDIDATE_ANTIGRAVITY,
            )
            self.assertEqual(record["private_run_count"], 2)
            self.assertEqual(record["official_evaluate_count"], 0)
            self.assertEqual(record["official_inspect_count"], 0)
            self.assertFalse(record["result_root_created"])
            self.assertIsNone(record["official_result"])
            self.assertIsNone(record["prediction_sha256"])
            self.assertIsNone(record["evaluation_sha256"])
            self.assertIsNone(record["decision_sha256"])
            self.assertIsNone(record["dashboard_snapshot_sha256"])
            self.assertFalse(record["test_labels_opened"])
            self.assertEqual(record["label_query_count"], 0)
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
            self.assertFalse(record["scientific_winner_selected"])
            self.assertFalse(record["official_root_present"])
            self.assertFalse(record["review_only"])
            self._assert_task5_result_antigravity(record["task5_private_train_result"])

        self.assertEqual(research_state["active_experiment"], "Exp464")
        self.assertEqual(research_state["active_phase"], STATUS_ANTIGRAVITY)
        self.assertEqual(research_state["status"], STATUS_ANTIGRAVITY)
        self.assertEqual(research_state["next_action"], EXPECTED_NEXT_ACTION_ANTIGRAVITY)
        self.assertEqual(
            research_state["exact_next_command"],
            EXPECTED_EXACT_NEXT_COMMAND_ANTIGRAVITY,
        )
        self.assertEqual(
            research_state["exact_next_command_if_interrupted"],
            EXPECTED_INTERRUPTED_COMMAND_ANTIGRAVITY,
        )
        self.assertEqual(
            research_state["official_readiness_command_after_fresh_sol_ship"],
            EXPECTED_OFFICIAL_COMMAND_ANTIGRAVITY,
        )
        self.assertEqual(
            research_state["last_verified_commit"],
            REVIEWED_COMMIT_ANTIGRAVITY,
        )
        self.assertEqual(research_state["pending_candidate"], EXPECTED_PENDING_ANTIGRAVITY)
        self.assertIn("retained for fresh review", research_state["pause_reason"])
        self.assertIn("Do not run Task 6", research_state["pause_reason"])

        for text in (
            state_record["notes"],
            catalog_record["notes"],
            champion_record["reason"],
        ):
            self.assertIn(REVIEWED_COMMIT_ANTIGRAVITY, text)
            self.assertIn(
                "exactly two sequential evaluate-to-inspect reproductions",
                text,
            )
            self.assertIn("0.8402557718424627 below 0.85", text)
            self.assertIn("Official evaluate/inspect counts remain 0", text)
            self.assertIn("Task 5 checkboxes remain unchecked and Task 6 is blocked", text)
            self.assertIn("reviewed-commit mismatch", text)
            self.assertIn("duplicate-process false positive", text)

        plan_text = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        self.assertIn(STATUS_ANTIGRAVITY, plan_text)
        self.assertIn(REVIEWED_COMMIT_ANTIGRAVITY, plan_text)
        self.assertIn("28/42", plan_text)
        self.assertIn("9/10", plan_text)
        self.assertIn("retained for fresh review", plan_text)
        self.assertIn("Task 5 checkboxes remain unchecked", plan_text)
        self.assertIn("Task 6 is blocked", plan_text)
        checkbox_parser = CheckboxParserAntigravity()
        checkbox_parser.feed(plan_text)
        self.assertEqual(checkbox_parser.checkbox_count, EXPECTED_CHECKBOX_COUNT_ANTIGRAVITY)
        self.assertEqual(checkbox_parser.checked_count, EXPECTED_CHECKED_COUNT_ANTIGRAVITY)
        task4_marker = '<h2><label><input type="checkbox" checked disabled> Task 4 —'
        task5_marker = '<h2><label><input type="checkbox" disabled> Task 5 —'
        task6_marker = '<h2><label><input type="checkbox" disabled> Task 6 —'
        task4_section = plan_text[plan_text.index(task4_marker):plan_text.index(task5_marker)]
        task5_section = plan_text[plan_text.index(task5_marker):plan_text.index(task6_marker)]
        task6_section = plan_text[plan_text.index(task6_marker):]
        self.assertEqual(task4_section.count('type="checkbox" checked disabled'), 7)
        self.assertEqual(task5_section.count('type="checkbox" checked disabled'), 0)
        self.assertEqual(task5_section.count('type="checkbox" disabled'), 7)
        self.assertEqual(task6_section.count('type="checkbox" checked disabled'), 0)
        self.assertEqual(task6_section.count('type="checkbox" disabled'), 7)

        parser = HtmlParserAntigravity()
        for relative in DOCUMENTS_ANTIGRAVITY + (PLAN_ANTIGRAVITY,):
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))
        parser.close()

        dashboard_text = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[0]).read_text(
            encoding="utf-8",
        )
        self.assertIn(STATUS_ANTIGRAVITY, dashboard_text)
        self.assertIn("정확히 두 번의 순차 evaluate→inspect reproduction", dashboard_text)
        self.assertIn(OFFICIAL_ROOT_ANTIGRAVITY, dashboard_text)
        self.assertIn("Task 6은 blocked", dashboard_text)

        preregistration_text = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[1]).read_text(
            encoding="utf-8",
        )
        self.assertIn("scientific contract stays frozen", preregistration_text)
        self.assertIn("private roots are retained for fresh review", preregistration_text)
        self.assertIn("private scientific run은 정확히 2회 완료", preregistration_text)

        runbook_text = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]).read_text(
            encoding="utf-8",
        )
        self.assertIn("exactly two private TRAIN reproductions already finished", runbook_text)
        self.assertIn("Do not run Task 6", runbook_text)
        self.assertIn("official root must stay absent", runbook_text)

        implementation_report_text = (
            ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]
        ).read_text(encoding="utf-8")
        self.assertIn(REVIEWED_COMMIT_ANTIGRAVITY, implementation_report_text)
        self.assertIn("exactly two private evaluate→inspect reproductions", implementation_report_text)
        self.assertIn("private-run result hashes", implementation_report_text)
        self.assertIn("selected backend: <code>numpy_cpu</code>", implementation_report_text)

        task4_report_text = (ROOT_ANTIGRAVITY / TASK4_REPORT_ANTIGRAVITY).read_text(
            encoding="utf-8",
        )
        for leaf in EXPECTED_LEAVES_ANTIGRAVITY:
            self.assertIn(leaf, task4_report_text)
        task5_report_text = (ROOT_ANTIGRAVITY / TASK5_REPORT_ANTIGRAVITY).read_text(
            encoding="utf-8",
        )
        self.assertIn(REVIEWED_COMMIT_ANTIGRAVITY, task5_report_text)
        self.assertIn("0.8402557718424627", task5_report_text)
        self.assertIn("exactly two sequential private non-official evaluate→inspect reproductions", task5_report_text)
        self.assertIn("fresh review", task5_report_text)

        builder = load_builder_module_antigravity()
        triplet_summary = builder.build_triplet_antigravity(validate_only=True)
        self.assertEqual(tuple(triplet_summary["triplet_paths"]), TRIPLET_ANTIGRAVITY)
        self.assertIn(
            triplet_summary["source_parent"],
            triplet_summary["allowed_source_commits"],
        )

        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        html_text = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(
            encoding="utf-8",
        )
        self.assertEqual(graph["graph_schema"], "exp464_task5_closed_cli_codemap_v1")
        self.assertEqual(lock["lock_schema"], "exp464_task5_closed_cli_codemap_lock_v1")
        self.assertEqual(tuple(graph["triplet_paths"]), TRIPLET_ANTIGRAVITY)
        self.assertEqual(lock["generated_files"], list(TRIPLET_ANTIGRAVITY))
        self.assertEqual(graph["source_parent"], lock["source_parent"])
        self.assertEqual(graph["source_parent"], triplet_summary["source_parent"])
        self.assertIn(graph["source_parent"], graph["allowed_source_commits"])
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
        self.assertIn(STATUS_ANTIGRAVITY, html_text)
        self.assertIn("Task 5 closed-CLI boundary code map", html_text)
        self.assertIn("두 번의 exact-equal private TRAIN reproduction truth", html_text)
        self.assertIn("internal official lifecycle authority", html_text)

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
