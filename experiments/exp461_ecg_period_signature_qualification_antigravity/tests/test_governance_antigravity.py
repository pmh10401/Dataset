from __future__ import annotations

import hashlib
import html.parser
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-20-exp461-ecg-period-signature-qualification-antigravity.html"
)
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/2026-08-20-exp461-ecg-period-signature-qualification-antigravity-design.html"
)
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP461_PREREGISTRATION_antigravity.html",
    "docs/EXP461_RUNBOOK_antigravity.html",
    "docs/EXP461_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP461_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp461_antigravity.html",
    "docs/codemap/codemap_exp461_antigravity.json",
    "docs/codemap/codemap_exp461_antigravity.lock",
)
RUN_ROOT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/exp461_ecg_period_signature_qualification_antigravity_20260820"
)
RUN_IDENTITY_ANTIGRAVITY = (
    "exp461-ecg-period-signature-qualification-antigravity-20260820"
)
PACKAGE_ANTIGRAVITY = "experiments/exp461_ecg_period_signature_qualification_antigravity"
DESIGN_SHA256_ANTIGRAVITY = (
    "3b1d6bc519a4acc000f45f30ba8ae5d610dcd0d2aa5e1cb61bd53837ba28f2c2"
)
PREREGISTRATION_SHA256_ANTIGRAVITY = (
    "9595d1f898b309ac899c821867a2f38e6f0eb24f147e6888970e1095534490cb"
)
RUNBOOK_SHA256_ANTIGRAVITY = (
    "3c9ce4c884ae0f5d7c9551b508a8e3092899f481fc3bc1a14adbd92f81f87807"
)
PENDING_CANDIDATE_ANTIGRAVITY = {
    "evaluation": None,
    "experiment": None,
    "prediction_sha256": None,
    "status": "none_exp446_publication_inspected_complete_not_pending_champion",
}
CHAMPION_ROOT_SNAPSHOT_ANTIGRAVITY = {
    "champion_status": "no_new_champion_verified_in_current_loop",
    "pending_candidate": PENDING_CANDIDATE_ANTIGRAVITY,
    "verified_references": [
        {
            "dataset_balanced_macro": {
                "f1": 0.6089825086931719,
                "false_positive_rate": 0.002196457972318159,
                "precision": 0.8262807700918903,
                "recall": 0.591448835046688,
            },
            "dataset_count": 1117,
            "evaluation_artifact": (
                "/Users/mac/Documents/Dataset/exp416_exp404_hard_test_confusion_audit/"
                "exp416_confusion_matrix.json"
            ),
            "micro": {
                "f1": 0.6241851368970013,
                "false_positive_rate": 0.0015782729630902897,
                "fn": 2003,
                "fp": 303,
                "precision": 0.8633904418394951,
                "recall": 0.4887697805002552,
                "tn": 191679,
                "tp": 1915,
            },
            "name": "Exp404 Hard prediction audited by Exp416",
            "operational_policy_changed": False,
            "prediction_coordinate_count": 2218,
            "prediction_dataset_count": 888,
            "prediction_sha256": (
                "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
            ),
            "role": "current_verified_full_population_reference_not_mandatory_code_base",
            "test_labels_loaded_after_prediction_freeze": True,
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


def map_builder_antigravity() -> object:
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp461_antigravity.py"
    if not path.is_file():
        raise AssertionError(f"missing map builder: {path}")
    spec = importlib.util.spec_from_file_location(
        "exp461_map_builder_antigravity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp461 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def test_task1_governance_antigravity(self) -> None:
        """Break caught: Task 1 planned truth drifts before scientific source work begins."""
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertTrue(path.is_file(), relative)
            self.assertIn("antigravity", path.name)
        for relative in DOCUMENTS_ANTIGRAVITY:
            parser = HtmlParserAntigravity()
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))
            parser.close()

        self.assertEqual(
            hashlib.sha256((ROOT_ANTIGRAVITY / DESIGN_ANTIGRAVITY).read_bytes()).hexdigest(),
            DESIGN_SHA256_ANTIGRAVITY,
        )

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)

        exp461_state = state["exp461"]
        catalog_entry = next(item for item in catalog["experiments"] if item["id"] == "Exp461")
        champion_entry = next(item for item in champion["history"] if item["experiment"] == "Exp461")
        self.assertEqual(sum(item["id"] == "Exp461" for item in catalog["experiments"]), 1)
        self.assertEqual(sum(item["experiment"] == "Exp461" for item in champion["history"]), 1)

        for record in (exp461_state, catalog_entry, champion_entry):
            with self.subTest(record=record["experiment"]):
                self.assertEqual(record["experiment"], "Exp461")
                self.assertEqual(record["kind"], "ecg_period_signature_qualification")
                self.assertEqual(record["status"], "planned_implementation_not_started")
                self.assertEqual(
                    record["implementation_status"],
                    "task_1_governance_complete_pending_external_review",
                )
                self.assertEqual(record["run_identity"], RUN_IDENTITY_ANTIGRAVITY)
                self.assertEqual(record["run_root"], RUN_ROOT_ANTIGRAVITY)
                self.assertEqual(record["package"], PACKAGE_ANTIGRAVITY)
                self.assertEqual(record["approved_design"], DESIGN_ANTIGRAVITY)
                self.assertEqual(record["approved_plan"], PLAN_ANTIGRAVITY)
                self.assertEqual(record["approved_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
                self.assertEqual(
                    record["documents"],
                    {
                        "preregistration": DOCUMENTS_ANTIGRAVITY[0],
                        "runbook": DOCUMENTS_ANTIGRAVITY[1],
                        "implementation_report": DOCUMENTS_ANTIGRAVITY[2],
                        "dashboard": DOCUMENTS_ANTIGRAVITY[3],
                    },
                )
                self.assertEqual(
                    record["predecessor"],
                    {
                        "experiment": "Exp460",
                        "truth_commit": "67d39f5c0a75e061e2d107fc9a9f58f52e76c2f1",
                        "artifact_map_sha256": (
                            "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916"
                        ),
                        "status": "fallback_triggered_performance_not_improved",
                        "failed_period_change_folds": [
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
                        ],
                    },
                )
                self.assertEqual(
                    record["scientific_contract"],
                    {
                        "target_dataset": "ECG5000_normal_2",
                        "data_split": "TRAIN_only",
                        "label_query_count_must_remain": 0,
                        "official_command_invocations_must_remain": 0,
                        "method_order": [
                            "hydra_local_gap",
                            "period_signature_local_gap",
                            "hydra_period_rank_max",
                        ],
                        "primary_candidate": "hydra_period_rank_max",
                        "seed_pair": [20260721, 20261721],
                        "blocked_fold_count": 5,
                        "per_type_auc_min": 0.85,
                        "pooled_auc_min": 0.90,
                        "normal_wilson_upper_max": 0.05,
                    },
                )
                self.assertFalse(record["official_evaluate_started"])
                self.assertFalse(record["official_inspect_started"])
                self.assertEqual(record["official_evaluate_invocations"], 0)
                self.assertEqual(record["official_inspect_invocations"], 0)
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
                self.assertFalse(record["review_only"])
                self.assertFalse(record["official_root_present"])

        self.assertEqual(state["pending_candidate"], PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(
            {
                "champion_status": champion["champion_status"],
                "pending_candidate": champion["pending_candidate"],
                "verified_references": champion["verified_references"],
            },
            CHAMPION_ROOT_SNAPSHOT_ANTIGRAVITY,
        )

    def test_task1_dashboard_plan_and_map_antigravity(self) -> None:
        """Break caught: Task 1 surface, checkboxes, or bounded map drift."""
        preregistration = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[0]).read_text(
            encoding="utf-8"
        )
        for phrase in (
            "planned-only governance",
            "planned_implementation_not_started",
            "Task 1/6",
            "official command count 0",
            "label query count 0",
            "result and artifact hashes",
            "null",
        ):
            self.assertIn(phrase, preregistration)

        runbook = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[1]).read_text(encoding="utf-8")
        for phrase in (
            "Task 1 operator boundary",
            RUN_ROOT_ANTIGRAVITY,
            "must stay absent",
            "Do not run official evaluate",
            "Do not run official inspect",
            "official command count 0",
        ):
            self.assertIn(phrase, runbook)

        report = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]).read_text(encoding="utf-8")
        for phrase in (
            "Task 4 qualification report",
            "SHA-256",
            DESIGN_SHA256_ANTIGRAVITY,
            PREREGISTRATION_SHA256_ANTIGRAVITY,
            RUNBOOK_SHA256_ANTIGRAVITY,
            "Task4 complete / Task5 next",
            "docs/EXP461_PREREGISTRATION_antigravity.html",
            "docs/EXP461_RUNBOOK_antigravity.html",
            "docs/EXP461_IMPLEMENTATION_REPORT_antigravity.html",
            "Task 2 handoff",
            "Task 2 scope truth",
            "Task 3 scope truth",
            "period_features_antigravity.py",
            "scoring_antigravity.py",
            "qualification_antigravity.py",
            "evaluation_antigravity.py",
            "artifacts_antigravity.py",
            "replay_antigravity.py",
            "dashboard_export_antigravity.py",
            "MLX smoke",
            "cpu_selected",
            "config_antigravity.py",
            "sources_antigravity.py",
        ):
            self.assertIn(phrase, report)
        for phrase in (
            "docs/superpowers/plans/2026-08-20-exp461-ecg-period-signature-qualification-antigravity.html",
            "docs/EXP461_DASHBOARD_antigravity.html",
            "docs/codemap/codemap_exp461_antigravity.json",
        ):
            self.assertNotIn(
                f"</code></td><td><code>{phrase}",
                report,
            )

        dashboard = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(encoding="utf-8")
        for phrase in (
            "Task 4/6",
            "planned_implementation_not_started",
            "Task4 complete / Task5 next",
            "공식 실행 0회",
            "label query count",
            "artifact map SHA-256",
            "null",
            "period_change",
            "20260721",
            "0.8208339620743423",
            "20261721",
            "0.8004124541019063",
            "hydra_local_gap",
            "period_signature_local_gap",
            "hydra_period_rank_max",
            "capture_exp460_period_authority_antigravity",
            "load_normal2_train_antigravity",
            "period_signature_antigravity",
            "anomaly_rank_antigravity",
            "score_fold_methods_antigravity",
            "qualify_methods_antigravity",
            "evaluate_qualification_antigravity",
            "publish_result_antigravity",
            "inspect_replay_at_antigravity",
            "export_dashboard_antigravity",
            "CPU NumPy",
        ):
            self.assertIn(phrase, dashboard)

        plan = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        task1 = re.search(
            r"<section class=\"task\">.*?<h2><label><input type=\"checkbox\"[^>]*>Task 1.*?</section>",
            plan,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(task1)
        assert task1 is not None
        tags = re.findall(r"<input\b[^>]*>", task1.group(0), flags=re.IGNORECASE)
        self.assertEqual(len(tags), 6)
        for tag in tags:
            self.assertRegex(tag, r"\bchecked(?:\s|=|>)")

        task2 = re.search(
            r"<section class=\"task\">\s*<h2><label><input type=\"checkbox\"[^>]*>Task 2.*?(?=<section class=\"task\">|<h2>완료 판단</h2>)",
            plan,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(task2)
        assert task2 is not None
        task2_tags = re.findall(r"<input\b[^>]*>", task2.group(0), flags=re.IGNORECASE)
        self.assertEqual(len(task2_tags), 6)
        for tag in task2_tags:
            self.assertRegex(tag, r"\bchecked(?:\s|=|>)")

        task3 = re.search(
            r"<section class=\"task\">\s*<h2><label><input type=\"checkbox\"[^>]*>Task 3.*?(?=<section class=\"task\">|<h2>완료 판단</h2>)",
            plan,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(task3)
        assert task3 is not None
        task3_tags = re.findall(r"<input\b[^>]*>", task3.group(0), flags=re.IGNORECASE)
        self.assertEqual(len(task3_tags), 7)
        for tag in task3_tags:
            self.assertRegex(tag, r"\bchecked(?:\s|=|>)")
            self.assertRegex(tag, r"\bdisabled(?:\s|=|>)")

        task4 = re.search(
            r"<section class=\"task\">\s*<h2><label><input type=\"checkbox\"[^>]*>Task 4.*?(?=<section class=\"task\">|<h2>완료 판단</h2>)",
            plan,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(task4)
        assert task4 is not None
        task4_tags = re.findall(r"<input\b[^>]*>", task4.group(0), flags=re.IGNORECASE)
        self.assertEqual(len(task4_tags), 7)
        for tag in task4_tags:
            self.assertRegex(tag, r"\bchecked(?:\s|=|>)")
            self.assertRegex(tag, r"\bdisabled(?:\s|=|>)")

        for task_number in range(5, 7):
            section = re.search(
                rf"<section class=\"task\">\s*<h2><label><input type=\"checkbox\"[^>]*>Task {task_number}.*?(?=<section class=\"task\">|<h2>완료 판단</h2>)",
                plan,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(section)
            assert section is not None
            later_tags = re.findall(r"<input\b[^>]*>", section.group(0), flags=re.IGNORECASE)
            self.assertGreaterEqual(len(later_tags), 2)
            for tag in later_tags:
                self.assertNotRegex(tag, r"\bchecked(?:\s|=|>)")

        builder = map_builder_antigravity()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict) and isinstance(lock, dict)
        self.assertLessEqual(len(graph["nodes"]), 40)
        self.assertEqual(graph["schema"], "exp461_task4_qualification_codemap_v1")
        self.assertEqual(lock["schema"], "exp461_task4_qualification_codemap_lock_v1")
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
            "task2_sources",
            "task2_config_tests",
            "task2_contracts_tests",
            "task2_predecessor_tests",
            "task2_sources_tests",
            "task3_period_features",
            "task3_scoring",
            "task3_period_features_tests",
            "task3_scoring_tests",
            "task4_qualification",
            "task4_evaluation",
            "task4_artifacts",
            "task4_replay",
            "task4_dashboard_export",
            "task4_qualification_tests",
            "task4_evaluation_tests",
            "task4_artifacts_tests",
            "task4_replay_tests",
            "task4_dashboard_export_tests",
        ):
            self.assertIn(node_id, node_ids)
        self.assertEqual(lock["represented_paths"], sorted(TRIPLET_ANTIGRAVITY))
        builder.validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    unittest.main()
