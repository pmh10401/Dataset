from __future__ import annotations

import hashlib
import html.parser
import importlib.util
import json
from pathlib import Path
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP458_PREREGISTRATION_antigravity.html",
    "docs/EXP458_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP458_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp458_antigravity.html",
    "docs/codemap/codemap_exp458_antigravity.json",
    "docs/codemap/codemap_exp458_antigravity.lock",
)
OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp458_ecg_normal2_rank_confidence_routing_antigravity_20260818"
)
PLAN_PATH_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-18-exp458-ecg-normal2-rank-confidence-routing-antigravity.html"
)
REVIEWED_COMMIT_ANTIGRAVITY = "2c836e371993b865b87d1b0b96595d0befd6ee3d"
TASK6_STATUS_ANTIGRAVITY = "routing_candidate_qualified_for_external_validation"
TASK6_LEDGER_STATUS_ANTIGRAVITY = (
    "official_evaluation_inspected_complete_qualified_for_external_validation"
)
TASK6_IMPLEMENTATION_STATUS_ANTIGRAVITY = "task_6_official_evaluation_and_inspection_complete"
TASK6_RUN_IDENTITY_ANTIGRAVITY = (
    "exp458-ecg-normal2-rank-confidence-routing-antigravity-20260818"
)
EXPECTED_RUNTIME_ANTIGRAVITY = {
    "executable": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
    "prefix": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314",
    "python_version": "3.14.6",
    "numpy_version": "2.3.5",
    "aeon_version": "1.4.0",
}
EXPECTED_RESULT_ANTIGRAVITY = {
    "tp": 617,
    "fp": 66,
    "fn": 829,
    "tn": 70788,
    "f1": 0.5796148426491311,
}
EXPECTED_OFFICIAL_RESULT_ANTIGRAVITY = {
    **EXPECTED_RESULT_ANTIGRAVITY,
    "passed": True,
    "status": TASK6_STATUS_ANTIGRAVITY,
}
EXPECTED_METHODS_ANTIGRAVITY = [
    {
        "method": "always_baseline",
        "tp": 609,
        "fp": 66,
        "fn": 837,
        "tn": 70788,
        "f1": 0.5742574257425742,
    },
    {
        "method": "family_wide_local_gap",
        "tp": 611,
        "fp": 63,
        "fn": 835,
        "tn": 70791,
        "f1": 0.5764150943396227,
    },
    {
        "method": "oracle_per_dataset",
        "tp": 618,
        "fp": 64,
        "fn": 828,
        "tn": 70790,
        "f1": 0.5808270676691729,
    },
    {
        "method": "normal2_rank_router",
        **EXPECTED_RESULT_ANTIGRAVITY,
    },
]
EXPECTED_CRITERIA_ANTIGRAVITY = [
    {
        "metric": "final_fn",
        "operator": "<=",
        "threshold": 834,
        "actual": 829,
        "passed": True,
    },
    {
        "metric": "final_fp",
        "operator": "<=",
        "threshold": 66,
        "actual": 66,
        "passed": True,
    },
    {
        "metric": "final_f1",
        "operator": ">",
        "threshold": 0.5764150943396227,
        "actual": 0.5796148426491311,
        "passed": True,
    },
]
EXPECTED_NORMAL2_ROUTE_ANTIGRAVITY = {
    "dataset": "ECG5000_normal_2",
    "rank_gate": True,
    "route": "candidate",
    "structure_gate": True,
}
EXPECTED_EFFECTIVE_ROUTES_ANTIGRAVITY = {
    "ChlorineConcentration_normal_1": "candidate",
    "ChlorineConcentration_normal_2": "candidate",
    "ChlorineConcentration_normal_3": "candidate",
    "ECG5000_normal_1": "baseline",
    "ECG5000_normal_2": "candidate",
    "ECG5000_normal_3": "baseline",
    "ECG5000_normal_4": "baseline",
    "ECG5000_normal_5": "baseline",
}
EXPECTED_LEAF_SHA256_ANTIGRAVITY = {
    "artifact_sha256_antigravity.json": (
        "7ef36f232db5d2f2396f3dcdeda15af2e93c34e18a6e05dc411e37fe855737f9"
    ),
    "comparison_metrics_antigravity.json": (
        "e62292db1796fdafe4b4fbbc2f7b57ce0d65b82c8812694eba24108df2092230"
    ),
    "context_antigravity.json": "b08051a5742479d6062367acd437683342d76e69a50faf8f7324a34b725c5f1d",
    "dashboard_snapshot_antigravity.json": (
        "9b4f95d11766076b23bfa4b670f1dbc9768f6bf8257f11aa5750768ff1fb8eb1"
    ),
    "decision_antigravity.json": "afcfc86cd0dbd9bcc08e3c59d3da7e256e82e58403b98736c7cdd448f597e6b0",
    "family_metrics_antigravity.json": (
        "f046424622cd419cb4e439f04b655850cdbe49f7b85553e3432332b800f5b2c6"
    ),
    "features_antigravity.jsonl": "bd0abbe0a086c994effb7aeae8b429919e414bc38069d6f71b49835ad62253b5",
    "route_decisions_antigravity.jsonl": (
        "d3f3469f509cc62f714ba274037cea6ba5a843d944a64a922523f6b98eaa78ed"
    ),
    "train_qualification_antigravity.jsonl": (
        "a770d2ddb3b5acd72ad370a4f7fc40072e6d31a50d9f6204e5cd72bd3f8da6e4"
    ),
}
EXPECTED_LIFECYCLE_ANTIGRAVITY = {
    "reviewed_commit": REVIEWED_COMMIT_ANTIGRAVITY,
    "evaluate_invocations": 1,
    "inspect_invocations": 1,
    "evaluate_exit_code": 0,
    "inspect_exit_code": 0,
    "result_root_created": True,
    "result_root_is_non_symlink_directory": True,
    "result_root_leaf_count": 9,
    "aggregate_json_equal": True,
    "semantic_replay_equal": True,
    "artifact_map_non_map_leaf_recomputation": "exact_match",
    "label_query_count": 0,
    "label_access_count": 0,
    "passed": True,
    "status": TASK6_STATUS_ANTIGRAVITY,
}
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


def jsonl_rows_antigravity(path: Path) -> list[object]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task_section_antigravity(plan: str, task_number: int) -> str:
    sections = plan.split("<section>")[1:]
    return sections[task_number - 1].split("</section>", 1)[0]


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def assert_task6_record_antigravity(self, record: dict[str, object]) -> None:
        self.assertEqual(record["experiment"], "Exp458")
        self.assertEqual(record["kind"], "ecg_normal2_rank_confidence_routing")
        self.assertEqual(record["status"], TASK6_LEDGER_STATUS_ANTIGRAVITY)
        self.assertEqual(record["implementation_status"], TASK6_IMPLEMENTATION_STATUS_ANTIGRAVITY)
        self.assertEqual(record["run_identity"], TASK6_RUN_IDENTITY_ANTIGRAVITY)
        self.assertTrue(record["official_evaluate_started"])
        self.assertTrue(record["official_inspect_started"])
        self.assertEqual(record["official_evaluate_invocations"], 1)
        self.assertEqual(record["official_inspect_invocations"], 1)
        self.assertTrue(record["result_root_created"])
        self.assertEqual(record["official_result"], EXPECTED_OFFICIAL_RESULT_ANTIGRAVITY)
        self.assertIsNone(record["prediction_sha256"])
        self.assertEqual(
            record["evaluation_sha256"],
            EXPECTED_LEAF_SHA256_ANTIGRAVITY["comparison_metrics_antigravity.json"],
        )
        self.assertEqual(
            record["decision_sha256"],
            EXPECTED_LEAF_SHA256_ANTIGRAVITY["decision_antigravity.json"],
        )
        self.assertEqual(
            record["artifact_map_sha256"],
            EXPECTED_LEAF_SHA256_ANTIGRAVITY["artifact_sha256_antigravity.json"],
        )
        self.assertEqual(
            record["dashboard_snapshot_sha256"],
            EXPECTED_LEAF_SHA256_ANTIGRAVITY["dashboard_snapshot_antigravity.json"],
        )
        self.assertFalse(record["routing_allowed"])
        self.assertFalse(record["promotion_allowed"])
        self.assertFalse(record["pending_candidate_changed"])
        self.assertFalse(record["scientific_winner_selected"])
        self.assertIs(record["test_labels_opened"], False)
        self.assertEqual(record["label_query_count"], 0)
        self.assertEqual(record["official_lifecycle"], EXPECTED_LIFECYCLE_ANTIGRAVITY)
        self.assertEqual(record["effective_routes"], EXPECTED_EFFECTIVE_ROUTES_ANTIGRAVITY)
        self.assertEqual(record["leaf_sha256"], EXPECTED_LEAF_SHA256_ANTIGRAVITY)

        private = record["private_candidate_verification"]
        self.assertEqual(private["runs"], 2)
        self.assertEqual(private["public_cli_evaluate_invocations"], 0)
        self.assertEqual(private["public_cli_inspect_invocations"], 0)
        self.assertTrue(private["official_root_absent_before_between_after"])
        self.assertTrue(private["temporary_roots_cleaned"])
        self.assertEqual(private["label_query_count_each_run"], [0, 0])
        self.assertFalse(private["official_hashes_remain_null"])

    def test_exp458_task6_official_result_truth_antigravity(self) -> None:
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertTrue(path.is_file(), relative)
            self.assertIn("antigravity", path.name)
        for relative in DOCUMENTS_ANTIGRAVITY:
            parser = HtmlParserAntigravity()
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))
            parser.close()

        self.assertTrue(OFFICIAL_ROOT_ANTIGRAVITY.is_dir())
        self.assertFalse(OFFICIAL_ROOT_ANTIGRAVITY.is_symlink())
        leaf_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(OFFICIAL_ROOT_ANTIGRAVITY.iterdir())
            if path.is_file() and not path.is_symlink()
        }
        self.assertEqual(leaf_hashes, EXPECTED_LEAF_SHA256_ANTIGRAVITY)
        artifact_map = strict_json_antigravity(
            OFFICIAL_ROOT_ANTIGRAVITY / "artifact_sha256_antigravity.json"
        )
        self.assertEqual(
            artifact_map,
            {
                key: value
                for key, value in EXPECTED_LEAF_SHA256_ANTIGRAVITY.items()
                if key != "artifact_sha256_antigravity.json"
            },
        )

        dashboard = strict_json_antigravity(
            OFFICIAL_ROOT_ANTIGRAVITY / "dashboard_snapshot_antigravity.json"
        )
        assert isinstance(dashboard, dict)
        self.assertEqual(dashboard["status"], TASK6_STATUS_ANTIGRAVITY)
        self.assertIs(dashboard["passed"], True)
        self.assertEqual(dashboard["methods"], EXPECTED_METHODS_ANTIGRAVITY)
        self.assertEqual(dashboard["criteria"], EXPECTED_CRITERIA_ANTIGRAVITY)
        self.assertEqual(dashboard["normal2_route"], EXPECTED_NORMAL2_ROUTE_ANTIGRAVITY)

        decision = strict_json_antigravity(OFFICIAL_ROOT_ANTIGRAVITY / "decision_antigravity.json")
        self.assertEqual(
            decision,
            {
                "status": TASK6_STATUS_ANTIGRAVITY,
                "passed": True,
                "criteria": EXPECTED_CRITERIA_ANTIGRAVITY,
            },
        )
        context = strict_json_antigravity(OFFICIAL_ROOT_ANTIGRAVITY / "context_antigravity.json")
        assert isinstance(context, dict)
        self.assertEqual(context["runtime"], EXPECTED_RUNTIME_ANTIGRAVITY)
        self.assertEqual(len(context["rows"]), 48)
        self.assertEqual(
            jsonl_rows_antigravity(OFFICIAL_ROOT_ANTIGRAVITY / "route_decisions_antigravity.jsonl"),
            [EXPECTED_NORMAL2_ROUTE_ANTIGRAVITY],
        )
        qualification_rows = jsonl_rows_antigravity(
            OFFICIAL_ROOT_ANTIGRAVITY / "train_qualification_antigravity.jsonl"
        )
        self.assertEqual(len(qualification_rows), 1)
        assert isinstance(qualification_rows[0], dict)
        self.assertEqual(qualification_rows[0]["dataset"], "ECG5000_normal_2")
        self.assertIs(qualification_rows[0]["passed"], True)
        self.assertEqual(
            [item["seed"] for item in qualification_rows[0]["seed_evidence"]],
            [20260721, 20261721],
        )

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)

        self.assertEqual(
            state["exp457"]["status"],
            "stopped_before_official_evaluation_train_gate_failed",
        )
        records = (
            state["exp458"],
            next(item for item in catalog["experiments"] if item["id"] == "Exp458"),
            next(item for item in champion["history"] if item["experiment"] == "Exp458"),
        )
        for record in records:
            assert isinstance(record, dict)
            self.assert_task6_record_antigravity(record)

        self.assertEqual(state["pending_candidate"], FROZEN_PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], FROZEN_PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(
            {key: champion[key] for key in FROZEN_CHAMPION_ANTIGRAVITY},
            FROZEN_CHAMPION_ANTIGRAVITY,
        )

    def test_dashboard_plan_and_map_record_task6_closeout_antigravity(self) -> None:
        dashboard_path = ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]
        report_path = ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[1]
        plan_path = ROOT_ANTIGRAVITY / PLAN_PATH_ANTIGRAVITY
        map_builder_path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp458_antigravity.py"
        graph_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1]
        lock_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2]
        rendered_path = ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]

        for path in (
            dashboard_path,
            report_path,
            plan_path,
            map_builder_path,
            graph_path,
            lock_path,
            rendered_path,
        ):
            self.assertTrue(path.is_file(), str(path.relative_to(ROOT_ANTIGRAVITY)))

        dashboard = dashboard_path.read_text(encoding="utf-8")
        for value in (
            "Task 6/6",
            TASK6_LEDGER_STATUS_ANTIGRAVITY,
            "공식 evaluate 1회 exit 0 / inspect 1회 exit 0 / 결과 폴더 leaf 9개",
            REVIEWED_COMMIT_ANTIGRAVITY,
            "TP 617 / FP 66 / FN 829 / TN 70788 / F1 0.5796148426491311",
            str(OFFICIAL_ROOT_ANTIGRAVITY),
            "artifact-map exact recomputation",
            "source-free semantic replay",
            "label_query_count는 0",
            "ECG5000_normal_2</code> candidate",
            "다른 ECG 네 개 baseline",
            "ChlorineConcentration 세 개 candidate",
            TASK6_STATUS_ANTIGRAVITY,
            "champion, pending candidate, routing, promotion, external dashboard publication은 바꾸지 않았다",
            EXPECTED_LEAF_SHA256_ANTIGRAVITY["artifact_sha256_antigravity.json"],
        ):
            self.assertIn(value, dashboard)

        report = report_path.read_text(encoding="utf-8")
        for value in (
            "Task 1부터 Task 6까지 완료했다",
            TASK6_LEDGER_STATUS_ANTIGRAVITY,
            "공식 evaluate 1회와 inspect 1회가 모두 exit 0",
            REVIEWED_COMMIT_ANTIGRAVITY,
            "Task 6 공식 lifecycle 결과",
            "leaf 9개",
            EXPECTED_LEAF_SHA256_ANTIGRAVITY["dashboard_snapshot_antigravity.json"],
            "독립 검증은 17개 check를 모두 통과했다",
            "label_query_count는 0",
            "champion, pending candidate, routing, promotion, operational alert 변경",
            "외부 검증 후보일 뿐 champion/promotion/deployment가 아니다",
        ):
            self.assertIn(value, report)

        plan = plan_path.read_text(encoding="utf-8")
        for task_number in range(1, 7):
            self.assertIn(
                f'<input type="checkbox" checked disabled>Task {task_number}:',
                plan,
            )
        self.assertNotIn('<input type="checkbox" disabled>', plan)
        expected_checked_counts = {1: 6, 2: 7, 3: 7, 4: 8, 5: 8, 6: 7}
        for task_number, checked_count in expected_checked_counts.items():
            section = task_section_antigravity(plan, task_number)
            self.assertEqual(
                section.count('<input type="checkbox" checked disabled>'),
                checked_count,
            )

        spec = importlib.util.spec_from_file_location(
            "exp458_map_builder_antigravity",
            map_builder_path,
        )
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        graph = strict_json_antigravity(graph_path)
        lock = strict_json_antigravity(lock_path)
        rendered = rendered_path.read_text(encoding="utf-8")
        assert isinstance(graph, dict)
        nodes = {node["id"]: node for node in graph["nodes"]}
        for node_id in (
            "approved_design",
            "implementation_plan",
            "preregistration",
            "implementation_report",
            "dashboard",
            "research_state",
            "experiment_catalog",
            "champion_ledger",
            "governance_regression",
            "map_builder",
            "frozen_config",
            "result_contracts",
            "rank_auc",
            "train_source",
            "train_qualification",
            "router",
            "evaluation",
            "artifacts",
            "replay",
            "dashboard_export",
            "cli",
            "cli_launcher",
        ):
            self.assertNotEqual(nodes[node_id].get("status"), "future_unknown")
            self.assertTrue((ROOT_ANTIGRAVITY / nodes[node_id]["path"]).exists())
        self.assertIn("Task 6 governance truth verification", nodes["governance_regression"]["role"])
        self.assertFalse(graph["unknown_relationships"])
        builder.validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    unittest.main()
