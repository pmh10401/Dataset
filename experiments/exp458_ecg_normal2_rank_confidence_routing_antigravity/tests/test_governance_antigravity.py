from __future__ import annotations

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
            "evaluation_artifact": "/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit/exp416_confusion_matrix.json",
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


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def test_exp458_is_planned_without_official_result_antigravity(self) -> None:
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertTrue(path.is_file(), relative)
            self.assertIn("antigravity", path.name)
        for relative in DOCUMENTS_ANTIGRAVITY:
            parser = HtmlParserAntigravity()
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))
            parser.close()

        self.assertFalse(OFFICIAL_ROOT_ANTIGRAVITY.exists())

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)

        exp457_state = state["exp457"]
        self.assertEqual(
            exp457_state["status"],
            "stopped_before_official_evaluation_train_gate_failed",
        )

        records = (
            state["exp458"],
            next(item for item in catalog["experiments"] if item["id"] == "Exp458"),
            next(item for item in champion["history"] if item["experiment"] == "Exp458"),
        )
        for record in records:
            self.assertEqual(record["experiment"], "Exp458")
            self.assertEqual(record["kind"], "ecg_normal2_rank_confidence_routing")
            self.assertEqual(
                record["status"],
                "implementation_verified_awaiting_official_evaluation",
            )
            self.assertEqual(
                record["implementation_status"],
                "task_5_private_train_candidate_verified_twice",
            )
            self.assertEqual(
                record["run_identity"],
                "exp458-ecg-normal2-rank-confidence-routing-antigravity-20260818",
            )
            self.assertEqual(
                record["approved_design"],
                "docs/superpowers/specs/2026-08-18-exp458-ecg-normal2-rank-confidence-routing-antigravity-design.html",
            )
            self.assertEqual(
                record["approved_design_sha256"],
                "52abe997c0e14badc2c886b888e3b2b2074cf002216891f173ab3b1dd6d7ea5d",
            )
            config = record["scientific_contract"]
            self.assertEqual(config["target_dataset"], "ECG5000_normal_2")
            self.assertEqual(config["hydra_seeds"], [20260721, 20261721])
            self.assertEqual(config["folds"], 5)
            self.assertEqual(
                config["primary_types"],
                [
                    "level_shift",
                    "gradual_drift",
                    "amplitude_scaling",
                    "period_change",
                ],
            )
            self.assertEqual(config["per_type_auc_min"], 0.85)
            self.assertEqual(config["pooled_auc_min"], 0.90)
            self.assertEqual(config["normal_wilson_upper_max"], 0.05)
            self.assertEqual(config["structure_gate"], "added_count > removed_count")
            self.assertEqual(
                config["projected_result"],
                {"tp": 617, "fp": 66, "fn": 829, "tn": 70788, "f1": 0.5796148426491311},
            )
            self.assertEqual(config["final_fn_max"], 834)
            self.assertEqual(config["final_fp_max"], 66)
            self.assertEqual(config["final_f1_strictly_above"], 0.5764150943396227)
            self.assertIs(config["test_metrics_may_inform_routing"], False)
            self.assertFalse(record["official_evaluate_started"])
            self.assertFalse(record["official_inspect_started"])
            self.assertEqual(record["official_evaluate_invocations"], 0)
            self.assertEqual(record["official_inspect_invocations"], 0)
            self.assertFalse(record["result_root_created"])
            self.assertIsNone(record["official_result"])
            for key in (
                "prediction_sha256",
                "evaluation_sha256",
                "decision_sha256",
                "artifact_map_sha256",
                "dashboard_snapshot_sha256",
            ):
                self.assertIsNone(record[key])
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
            self.assertFalse(record["scientific_winner_selected"])
            self.assertIs(record["test_labels_opened"], False)
            self.assertEqual(record["label_query_count"], 0)
            private = record["private_candidate_verification"]
            self.assertEqual(private["runs"], 2)
            self.assertEqual(private["public_cli_evaluate_invocations"], 0)
            self.assertEqual(private["public_cli_inspect_invocations"], 0)
            self.assertTrue(private["official_root_absent_before_between_after"])
            self.assertTrue(private["temporary_roots_cleaned"])
            self.assertEqual(private["label_query_count_each_run"], [0, 0])
            self.assertEqual(
                private["candidate_payload_sha256"],
                "0150a4243a287131adf433f0f04503974440f88cb79cf6a14e18257e55226fa8",
            )
            self.assertTrue(private["leaf_sha256_equal"])
            self.assertTrue(private["dashboard_equal"])
            self.assertTrue(private["qualification_equal"])
            self.assertTrue(private["route_equal"])
            self.assertEqual(private["route"], "candidate")
            self.assertEqual(
                private["result"],
                {"tp": 617, "fp": 66, "fn": 829, "tn": 70788, "f1": 0.5796148426491311},
            )
            self.assertEqual(
                private["criteria_passed"],
                ["final_fn", "final_fp", "final_f1"],
            )
            self.assertTrue(private["official_hashes_remain_null"])
        self.assertEqual(state["pending_candidate"], FROZEN_PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], FROZEN_PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(
            {key: champion[key] for key in FROZEN_CHAMPION_ANTIGRAVITY},
            FROZEN_CHAMPION_ANTIGRAVITY,
        )

    def test_dashboard_plan_and_map_stay_truthful_antigravity(self) -> None:
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
            "implementation_verified_awaiting_official_evaluation",
            "AUC",
            "ECG5000_normal_2",
            "상위 0.5%만 이상으로 인정한 지나치게 극단적인 통과선",
            "각 합성 이상 AUC",
            "모두 ≥ 0.85",
            "평균 AUC",
            "≥ 0.90",
            "14&gt;5",
            "TP 617 / FP 66 / FN 829 / TN 70788 / F1 0.5796148426491311",
            "공식 결과가 아니라 두 번 재현한 TRAIN 후보 결과",
            "Task 5/6",
            "공식 evaluate 0회 / inspect 0회 / 결과 폴더 없음",
            "0150a4243a287131adf433f0f04503974440f88cb79cf6a14e18257e55226fa8",
            "두 seed 20260721와 20261721가 각각 모든 gate를 만족해야 한다.",
            "Task 5 closed CLI",
        ):
            self.assertIn(value, dashboard)

        report = report_path.read_text(encoding="utf-8")
        self.assertIn("implementation_verified_awaiting_official_evaluation", report)
        self.assertIn("Task 1부터 Task 5까지 완료했다", report)
        self.assertIn("공식 evaluate/inspect, TEST label", report)
        self.assertIn("6e8e500d19afcb17907ef53ad1527b5781c6d007", report)
        self.assertIn("0.9796866524522819", report)
        self.assertIn("0.9485180577280324", report)
        self.assertIn("Task4 uses synthetic/frozen evidence only", report)
        self.assertIn("actual TRAIN candidate verification을 정확히 두 번 실행했다", report)
        self.assertIn("0150a4243a287131adf433f0f04503974440f88cb79cf6a14e18257e55226fa8", report)

        plan = plan_path.read_text(encoding="utf-8")
        self.assertIn('<input type="checkbox" checked disabled>Task 1:', plan)
        self.assertIn('<input type="checkbox" checked disabled>Task 2:', plan)
        self.assertIn('<input type="checkbox" checked disabled>Task 3:', plan)
        task1 = plan.split("<section>", 1)[1].split("</section>", 1)[0]
        self.assertEqual(task1.count('<input type="checkbox" checked disabled>'), 6)
        task2 = plan.split("Task 2:", 1)[1].split("Task 3:", 1)[0]
        self.assertEqual(task2.count('<input type="checkbox" checked disabled>'), 7)
        task3 = plan.split("Task 3:", 1)[1].split("Task 4:", 1)[0]
        self.assertEqual(task3.count('<input type="checkbox" checked disabled>'), 7)
        task4 = plan.split("Task 4:", 1)[1].split("</section>", 1)[0]
        self.assertEqual(task4.count('<input type="checkbox" checked disabled>'), 7)
        task5 = plan.split("Task 5:", 1)[1].split("</section>", 1)[0]
        self.assertEqual(task5.count('<input type="checkbox" checked disabled>'), 7)
        for task_number in range(2, 7):
            if task_number in {2, 3, 4, 5}:
                continue
            task = plan.split(f"Task {task_number}:", 1)[1]
            if task_number < 6:
                task = task.split(f"Task {task_number + 1}:", 1)[0]
            else:
                task = task.split("</section>", 1)[0]
            self.assertEqual(task.count('<input type="checkbox" checked disabled>'), 0)

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
        self.assertFalse(graph["unknown_relationships"])
        builder.validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    unittest.main()
