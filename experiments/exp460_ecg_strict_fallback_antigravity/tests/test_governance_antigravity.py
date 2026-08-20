from __future__ import annotations

import html.parser
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-20-exp460-ecg-strict-fallback-antigravity.html"
)
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP460_PREREGISTRATION_antigravity.html",
    "docs/EXP460_RUNBOOK_antigravity.html",
    "docs/EXP460_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP460_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp460_antigravity.html",
    "docs/codemap/codemap_exp460_antigravity.json",
    "docs/codemap/codemap_exp460_antigravity.lock",
)
FROZEN_PENDING_CANDIDATE_ANTIGRAVITY = {
    "experiment": None,
    "status": "none_exp446_publication_inspected_complete_not_pending_champion",
    "prediction_sha256": None,
    "evaluation": None,
}
FROZEN_CHAMPION_ANTIGRAVITY = {
    "champion_status": "no_new_champion_verified_in_current_loop",
    "verified_references": [
        {
            "name": "Exp404 Hard prediction audited by Exp416",
            "role": "current_verified_full_population_reference_not_mandatory_code_base",
            "prediction_sha256": "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
            "evaluation_artifact": (
                "/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit/"
                "exp416_confusion_matrix.json"
            ),
            "dataset_count": 1117,
            "prediction_coordinate_count": 2218,
            "prediction_dataset_count": 888,
            "micro": {
                "tp": 1915,
                "fp": 303,
                "fn": 2003,
                "tn": 191679,
                "precision": 0.8633904418394951,
                "recall": 0.4887697805002552,
                "f1": 0.6241851368970013,
                "false_positive_rate": 0.0015782729630902897,
            },
            "dataset_balanced_macro": {
                "precision": 0.8262807700918903,
                "recall": 0.591448835046688,
                "f1": 0.6089825086931719,
                "false_positive_rate": 0.002196457972318159,
            },
            "test_labels_loaded_after_prediction_freeze": True,
            "operational_policy_changed": False,
        }
    ],
}
EXPECTED_RULE_ORDER_ANTIGRAVITY = [
    "always_baseline",
    "always_candidate",
    "majority_8_of_10",
    "strict_10_of_10",
]
EXPECTED_FAILED_FOLDS_ANTIGRAVITY = [
    {
        "seed": 20260721,
        "fold_index": 3,
        "period_change_auc": 0.8208339620743423,
    },
    {
        "seed": 20261721,
        "fold_index": 4,
        "period_change_auc": 0.8004124541019063,
    },
]
EXPECTED_PRIVATE_VALIDATION_ANTIGRAVITY = {
    "run_count": 2,
    "payload_exact_equal": True,
    "leaf_hashes_exact_equal": True,
    "leaf_count": 6,
    "temporary_roots_cleaned": True,
    "official_root_absent": True,
    "passed_record_count": 8,
    "failed_fold_count": 2,
    "worst_period_change_auc": 0.8004124541019063,
    "strict_route": "baseline",
    "status": "fallback_triggered_performance_not_improved",
    "joint_success": False,
}
EXPECTED_OFFICIAL_RESULT_ANTIGRAVITY = {
    "reviewed_commit": "ca51813450f26b8a3fada6f39cf810452467297e",
    "status": "fallback_triggered_performance_not_improved",
    "strict_route": "baseline",
    "joint_success": False,
    "passed_record_count": 8,
    "failed_fold_count": 2,
    "worst_period_change_auc": 0.8004124541019063,
    "leaf_sha256": {
        "context_antigravity.json": "1c30f92d60ca76214f680c4452b1a4418764da84fe5ec1f8a5156f743b84db9f",
        "rule_comparisons_antigravity.jsonl": "3ed3f05f5359cbe4d1a0dbcad371bb3ed951679c1b153d59ff591933fc13dc75",
        "failed_folds_antigravity.jsonl": "880d67769d8e4e40b3b504a0573878bdb31a364a33fdd48d70af0fe2b2a7b198",
        "decision_antigravity.json": "0859be73fb8b7c77cc803d67745ecd1bd42766ea5e3c005ce1bb9d2d3e6342c1",
        "dashboard_snapshot_antigravity.json": "fea8b4e43d256feddef48545e201f674bfe39fe85559050ed405f4e996709ee5",
        "artifact_sha256_antigravity.json": "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916",
    },
}


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates_antigravity,
    )


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def test_task6_governance_surface_antigravity(self) -> None:
        """Break caught: Task 6 official truth, frozen result, or unchanged routing state drifts."""
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertTrue(path.is_file(), relative)
            self.assertIn("antigravity", path.name)
        for relative in DOCUMENTS_ANTIGRAVITY:
            parser = HtmlParserAntigravity()
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))
            parser.close()

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)
        records = (
            state["exp460"],
            next(item for item in catalog["experiments"] if item["id"] == "Exp460"),
            next(item for item in champion["history"] if item["experiment"] == "Exp460"),
        )
        for record in records:
            self.assertEqual(record["kind"], "ecg_strict_fallback")
            self.assertEqual(
                record["status"],
                "official_evaluation_inspected_complete_fallback_no_performance_improvement",
            )
            self.assertEqual(
                record["implementation_status"],
                "task_6_official_evaluation_inspected_complete",
            )
            contract = record["scientific_contract"]
            self.assertEqual(contract["target_dataset"], "ECG5000_normal_2")
            self.assertEqual(contract["rule_order"], EXPECTED_RULE_ORDER_ANTIGRAVITY)
            self.assertEqual(contract["official_decision_rule"], "strict_10_of_10_only")
            self.assertEqual(contract["strict_failure_route"], "baseline")
            self.assertEqual(contract["failed_folds"], EXPECTED_FAILED_FOLDS_ANTIGRAVITY)
            self.assertEqual(contract["exp459_status"], "unstable")
            self.assertTrue(record["official_evaluate_started"])
            self.assertTrue(record["official_inspect_started"])
            self.assertEqual(record["official_evaluate_invocations"], 1)
            self.assertEqual(record["official_inspect_invocations"], 1)
            self.assertTrue(record["result_root_created"])
            self.assertEqual(
                record["run_root"],
                "/Users/mac/Documents/Dataset/exp460_ecg_strict_fallback_antigravity_20260820",
            )
            self.assertEqual(record["official_result"], EXPECTED_OFFICIAL_RESULT_ANTIGRAVITY)
            self.assertIsNone(record["prediction_sha256"])
            self.assertEqual(record["evaluation_sha256"], "3ed3f05f5359cbe4d1a0dbcad371bb3ed951679c1b153d59ff591933fc13dc75")
            self.assertEqual(record["decision_sha256"], "0859be73fb8b7c77cc803d67745ecd1bd42766ea5e3c005ce1bb9d2d3e6342c1")
            self.assertEqual(record["artifact_map_sha256"], "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916")
            self.assertEqual(record["dashboard_snapshot_sha256"], "fea8b4e43d256feddef48545e201f674bfe39fe85559050ed405f4e996709ee5")
            self.assertFalse(record["test_labels_opened"])
            self.assertEqual(record["label_query_count"], 0)
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
            self.assertFalse(record["scientific_winner_selected"])
            self.assertEqual(
                record["private_validation"],
                EXPECTED_PRIVATE_VALIDATION_ANTIGRAVITY,
            )
        self.assertEqual(state["pending_candidate"], FROZEN_PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], FROZEN_PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(
            {key: champion[key] for key in FROZEN_CHAMPION_ANTIGRAVITY},
            FROZEN_CHAMPION_ANTIGRAVITY,
        )

    def test_task6_plan_dashboard_report_runbook_and_map_antigravity(self) -> None:
        """Break caught: Task 6 result closeout, completed steps, or bounded map drifts."""
        dashboard = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(encoding="utf-8")
        for value in (
            "Task 6/6",
            "official_evaluation_inspected_complete_fallback_no_performance_improvement",
            "official evaluate/inspect counts",
            "1/1",
            "label query count",
            "frozen predecessor authority",
            "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
            "a30fbe7ed34f88c1ab9b7fd40963edebbf1a4127d8d64c99141224cb2dbb106d",
            "527318cacdbc63905ab58237d793803eb286192456eb007362e6bed11c8f8716",
            "strict_10_of_10",
            "always_baseline",
            "always_candidate",
            "majority_8_of_10",
            "strict final status",
            "fallback_triggered_performance_not_improved",
            "joint success",
            "false",
            "Task 6 official lifecycle",
            "Current authority summary",
            "Strict route outcome",
            "local artifact inventory",
            "context_antigravity.json",
            "rule_comparisons_antigravity.jsonl",
            "failed_folds_antigravity.jsonl",
            "decision_antigravity.json",
            "dashboard_snapshot_antigravity.json",
            "artifact_sha256_antigravity.json",
            "payload exact-equal",
            "leaf hashes exact-equal",
            "private roots cleaned",
            "exact fixed interpreter",
            "evaluate --reviewed-commit LOWER40HEX",
            "bare inspect",
            "20260721",
            "0.8208339620743423",
            "20261721",
            "0.8004124541019063",
            "617",
            "829",
            "0.5796148426491311",
            "609",
            "837",
            "0.5742574257425742",
            "result and artifact hashes",
            "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916",
            "ca51813450f26b8a3fada6f39cf810452467297e",
            "wrapper exit 127",
        ):
            self.assertIn(value, dashboard)

        report = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]).read_text(encoding="utf-8")
        for value in (
            "Task 6 official lifecycle",
            "Task 6/6",
            "FAILED (failures=4, errors=2)",
            "Ran 6 tests",
            "Ran 31 tests",
            "Ran 57 tests",
            "FAILED (errors=1)",
            "fallback_triggered_performance_not_improved",
            "8/10 pass",
            "majority route",
            "candidate",
            "strict route",
            "baseline",
            "semantic replay",
            "bounded dashboard payload",
            "Exp459 historical governance note",
            "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916",
            "private roots cleaned",
            "official root present",
            "evaluate exit 0",
            "inspect exit 0",
        ):
            self.assertIn(value, report)

        runbook = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[1]).read_text(encoding="utf-8")
        for value in (
            "Task 6 terminal operating boundary",
            "official_evaluation_inspected_complete_fallback_no_performance_improvement",
            "evaluate --reviewed-commit LOWER40HEX",
            "bare inspect",
            "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
            "/Users/mac/Documents/Dataset/exp460_ecg_strict_fallback_antigravity_20260820",
            "Task 6",
            "official evaluate/inspect counts 1/1",
            "label query count 0",
            "result hashes frozen",
        ):
            self.assertIn(value, runbook)

        plan = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        self.assertIn('<h2><label><input type="checkbox" checked disabled>Task 1', plan)
        task1 = plan.split("Task 1", 1)[1].split("Task 2", 1)[0]
        self.assertEqual(
            task1.count('<label><input type="checkbox" checked disabled>Step '),
            6,
        )
        for step_label in (
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Step 5",
            "Step 6",
        ):
            self.assertIn(
                f'<label><input type="checkbox" checked disabled>{step_label}',
                task1,
            )
        self.assertIn('<h2><label><input type="checkbox" checked disabled>Task 2', plan)
        task2 = plan.split("Task 2", 1)[1].split("Task 3", 1)[0]
        self.assertEqual(
            task2.count('<label><input type="checkbox" checked disabled>Step '),
            6,
        )
        for step_label in (
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Step 5",
            "Step 6",
        ):
            self.assertIn(
                f'<label><input type="checkbox" checked disabled>{step_label}',
                task2,
            )
        self.assertIn('<h2><label><input type="checkbox" checked disabled>Task 3', plan)
        task3 = plan.split("Task 3", 1)[1].split("Task 4", 1)[0]
        self.assertEqual(
            task3.count('<label><input type="checkbox" checked disabled>Step '),
            6,
        )
        for step_label in (
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Step 5",
            "Step 6",
        ):
            self.assertIn(
                f'<label><input type="checkbox" checked disabled>{step_label}',
                task3,
            )
        self.assertIn('<h2><label><input type="checkbox" checked disabled>Task 4', plan)
        task4 = plan.split("Task 4", 1)[1].split("Task 5", 1)[0]
        self.assertEqual(
            task4.count('<label><input type="checkbox" checked disabled>Step '),
            6,
        )
        for step_label in (
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Step 5",
            "Step 6",
        ):
            self.assertIn(
                f'<label><input type="checkbox" checked disabled>{step_label}',
                task4,
            )
        self.assertIn(
            '<h2><label><input type="checkbox" checked disabled>Task 5',
            plan,
        )
        task5 = plan.split("Task 5", 1)[1].split("Task 6", 1)[0]
        self.assertEqual(
            task5.count('<label><input type="checkbox" checked disabled>Step '),
            6,
        )
        for step_label in (
            "Step 1",
            "Step 2",
            "Step 3",
            "Step 4",
            "Step 5",
            "Step 6",
        ):
            self.assertIn(
                f'<label><input type="checkbox" checked disabled>{step_label}',
                task5,
            )
        self.assertIn('<h2>Task 6', plan)
        self.assertNotIn(
            '<h2><label><input type="checkbox" checked disabled>Task 6',
            plan,
        )
        task6 = plan.split("<h2>Task 6", 1)[1].split("완료 판단", 1)[0]
        self.assertEqual(
            task6.count('<label><input type="checkbox" checked disabled>Step '),
            6,
        )
        self.assertIn('<label><input type="checkbox" disabled>Step 7', task6)
        self.assertIn("HTTP 402 usage balance exhausted", task6)

        spec = importlib.util.spec_from_file_location(
            "exp460_map_builder_antigravity",
            ROOT_ANTIGRAVITY / "scratch/build_codemap_exp460_antigravity.py",
        )
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict)
        self.assertLessEqual(len(graph["nodes"]), 25)
        self.assertEqual(graph["schema"], "exp460_task6_official_fallback_codemap_v1")
        self.assertEqual(lock["schema"], "exp460_task6_official_fallback_codemap_lock_v1")
        node_ids = {node["id"] for node in graph["nodes"]}
        for node_id in (
            "approved_design",
            "implementation_plan",
            "preregistration",
            "runbook",
            "implementation_report",
            "dashboard",
            "research_state",
            "experiment_catalog",
            "champion_ledger",
            "governance_regression",
            "map_builder",
            "task2_config",
            "task2_contracts",
            "task2_predecessor_capture",
            "task2_config_tests",
            "task2_contracts_tests",
            "task2_predecessor_tests",
            "task3_strict_route",
            "task3_rule_evaluation",
            "task4_artifacts_publication",
            "task4_replay",
            "task4_dashboard_export",
            "task5_cli",
            "task5_launcher",
            "task5_cli_tests",
        ):
            self.assertIn(node_id, node_ids)
        self.assertFalse(
            any(node["id"].startswith("future_") for node in graph["nodes"])
        )
        builder.validate_map_antigravity(graph, lock, rendered)

    def test_validate_only_reads_stored_triplet_and_rejects_stale_bytes_antigravity(self) -> None:
        """Break caught: validate-only ignores the packaged triplet and validates rebuilt bytes instead."""
        command = [
            "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
            "scratch/build_codemap_exp460_antigravity.py",
            "--validate-only",
        ]
        success = subprocess.run(
            command,
            cwd=ROOT_ANTIGRAVITY,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(success.returncode, 0, success.stderr)

        html_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]
        original = html_path.read_text(encoding="utf-8")
        mutated = original.replace("Task 6 official fallback 코드맵", "Task 6 official fallback 코드맵 tampered", 1)
        self.assertNotEqual(mutated, original)
        html_path.write_text(mutated, encoding="utf-8")
        try:
            failure = subprocess.run(
                command,
                cwd=ROOT_ANTIGRAVITY,
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            html_path.write_text(original, encoding="utf-8")
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("rendered HTML is not in sync with graph JSON", failure.stderr)


if __name__ == "__main__":
    unittest.main()
