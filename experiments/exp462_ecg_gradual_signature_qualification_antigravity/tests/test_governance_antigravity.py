from __future__ import annotations

import hashlib
import html
import html.parser
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity.html"
)
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/2026-08-20-exp462-ecg-gradual-signature-qualification-antigravity-design.html"
)
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP462_PREREGISTRATION_antigravity.html",
    "docs/EXP462_RUNBOOK_antigravity.html",
    "docs/EXP462_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP462_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp462_antigravity.html",
    "docs/codemap/codemap_exp462_antigravity.json",
    "docs/codemap/codemap_exp462_antigravity.lock",
)
RUN_ROOT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/exp462_ecg_gradual_signature_qualification_antigravity_20260820"
)
RUN_IDENTITY_ANTIGRAVITY = (
    "exp462-ecg-gradual-signature-qualification-antigravity-20260820"
)
PACKAGE_ANTIGRAVITY = "experiments/exp462_ecg_gradual_signature_qualification_antigravity"
DESIGN_SHA256_ANTIGRAVITY = (
    "34a950e619b7ddd1accc7cc41281a476611010f27ef02b1a73fa46523d064829"
)
PENDING_CANDIDATE_ANTIGRAVITY = {
    "evaluation": None,
    "experiment": None,
    "prediction_sha256": None,
    "status": "none_exp446_publication_inspected_complete_not_pending_champion",
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
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp462_antigravity.py"
    if not path.is_file():
        raise AssertionError(f"missing map builder: {path}")
    spec = importlib.util.spec_from_file_location(
        "exp462_map_builder_antigravity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp462 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def test_task1_planned_governance_antigravity(self) -> None:
        """Break caught: Task 1 planned truth drifts into result or official lifecycle work."""
        missing = [
            relative
            for relative in DOCUMENTS_ANTIGRAVITY[:3] + TRIPLET_ANTIGRAVITY
            if not (ROOT_ANTIGRAVITY / relative).is_file()
        ]
        self.assertEqual(missing, [], f"missing Task 1 artifacts: {missing}")
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
        run_root = Path(RUN_ROOT_ANTIGRAVITY)
        self.assertFalse(run_root.exists(), RUN_ROOT_ANTIGRAVITY)
        self.assertFalse(run_root.is_symlink(), RUN_ROOT_ANTIGRAVITY)

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(catalog, dict) and isinstance(champion, dict)

        exp462_state = state["exp462"]
        catalog_entry = next(item for item in catalog["experiments"] if item["id"] == "Exp462")
        champion_entry = next(item for item in champion["history"] if item["experiment"] == "Exp462")
        self.assertEqual(sum(item["id"] == "Exp462" for item in catalog["experiments"]), 1)
        self.assertEqual(sum(item["experiment"] == "Exp462" for item in champion["history"]), 1)

        for record in (exp462_state, catalog_entry, champion_entry):
            with self.subTest(record=record["experiment"]):
                self.assertEqual(record["experiment"], "Exp462")
                self.assertEqual(record["kind"], "ecg_gradual_signature_qualification")
                self.assertEqual(record["status"], "planned_implementation_not_started")
                self.assertEqual(record["implementation_status"], "task_1_complete_pending_task_2")
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
                        "experiment": "Exp461",
                        "truth_commit": "3a306ff3e276dae13b28b9733db30e9934da7eb2",
                        "artifact_map_sha256": (
                            "8d0992d362cefab305ef8801afd9d1f5d51522e623fa372ba18f6d5d0f3478be"
                        ),
                        "decision_sha256": (
                            "27f6d8db82b71051a16b68d3ad2f69f6608f6a922d765b35029e594afc581315"
                        ),
                        "status": "qualification_failed_before_official_evaluation",
                        "primary_candidate": "hydra_period_rank_max",
                        "primary_passed_count": 9,
                        "primary_required_count": 10,
                        "failed_gradual_drift_auc": 0.8402306222021025,
                        "period_min_auc": 0.907392736783864,
                        "historical_period_folds": [
                            {
                                "seed": 20260721,
                                "fold_index": 3,
                                "period_change_auc": 0.907392736783864,
                            },
                            {
                                "seed": 20261721,
                                "fold_index": 4,
                                "period_change_auc": 0.9216148584075248,
                            },
                        ],
                    },
                )
                self.assertEqual(
                    record["scientific_contract"],
                    {
                        "target_dataset": "ECG5000_normal_2",
                        "data_split": "TRAIN_only",
                        "train_shape": [1413, 140],
                        "label_query_count_must_remain": 0,
                        "official_command_invocations_must_remain": 0,
                        "method_order": [
                            "hydra_period_rank_max",
                            "gradual_signature_local_gap",
                            "hydra_period_gradual_rank_max",
                        ],
                        "primary_candidate": "hydra_period_gradual_rank_max",
                        "seed_pair": [20260721, 20261721],
                        "blocked_fold_count": 5,
                        "per_type_auc_min": 0.85,
                        "pooled_auc_min": 0.90,
                        "normal_wilson_upper_max": 0.05,
                        "gradual_drift_auc_min": 0.85,
                        "period_change_auc_global_min": 0.907392736783864,
                        "historical_period_fold_thresholds": [
                            {
                                "seed": 20260721,
                                "fold_index": 3,
                                "period_change_auc_min": 0.907392736783864,
                            },
                            {
                                "seed": 20261721,
                                "fold_index": 4,
                                "period_change_auc_min": 0.9216148584075248,
                            },
                        ],
                        "test_metrics_may_inform_routing": False,
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
                self.assertFalse(record["official_root_present"])
                self.assertTrue(record["review_only"])

        self.assertEqual(state["pending_candidate"], PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["champion_status"], "no_new_champion_verified_in_current_loop")

    def test_task1_dashboard_plan_and_map_antigravity(self) -> None:
        """Break caught: Task 1 surface, checkbox state, or bounded map drifts."""
        missing = [
            relative
            for relative in DOCUMENTS_ANTIGRAVITY[:3] + TRIPLET_ANTIGRAVITY
            if not (ROOT_ANTIGRAVITY / relative).is_file()
        ]
        self.assertEqual(missing, [], f"missing Task 1 artifacts: {missing}")
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
            "gradual_drift AUC ≥ 0.85",
            "period_change AUC ≥ 0.907392736783864",
        ):
            self.assertIn(phrase, preregistration)

        runbook = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[1]).read_text(encoding="utf-8")
        for phrase in (
            "Task 1 governance ready / Task 2 next",
            RUN_ROOT_ANTIGRAVITY,
            "must stay absent",
            "Do not run official evaluate",
            "Do not run official inspect",
            "official command count 0",
            "label query count 0",
        ):
            self.assertIn(phrase, runbook)

        report = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]).read_text(encoding="utf-8")
        for phrase in (
            "Exp462 implementation report",
            "planned_implementation_not_started",
            "Task 2 implemented pending review / Task 3 not started",
            "hydra_period_gradual_rank_max",
            "0.8402306222021025",
            "0.907392736783864",
            "0.9216148584075248",
            "official command count 0",
            "label query count 0",
            "artifact map null",
            "capture_exp461_gradual_authority_antigravity()",
            "load_normal2_train_antigravity()",
        ):
            self.assertIn(phrase, report)

        dashboard = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(encoding="utf-8")
        for phrase in (
            "Task 2 implemented pending review / Task 3 not started",
            "planned_implementation_not_started",
            "task_2_implemented_pending_review",
            "official evaluate 0 / official inspect 0",
            "label query count",
            "official artifact map SHA-256",
            "null",
            "Exp461 failure 9/10",
            "0340b6ee737da84777283668449738d0c1585566",
            "capture_exp461_gradual_authority_antigravity()",
            "load_normal2_train_antigravity()",
            "gradual_drift",
            "0.8402306222021025",
            "0.85",
            "period_change",
            "0.907392736783864",
            "0.9216148584075248",
            "hydra_period_rank_max",
            "gradual_signature_local_gap",
            "hydra_period_gradual_rank_max",
        ):
            self.assertIn(phrase, dashboard)

        plan = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        self.assertIn("Task 2 implemented pending review / Task 3 not started", plan)
        for task_number in range(1, 7):
            task = re.search(
                rf"<section>\s*<h2><label><input type=\"checkbox\"[^>]*>Task {task_number}.*?</section>",
                plan,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(task, f"Task {task_number} section missing")
            assert task is not None
            tags = re.findall(r"<input\b[^>]*>", task.group(0), flags=re.IGNORECASE)
            self.assertGreaterEqual(len(tags), 1, f"Task {task_number} inputs missing")
            if task_number == 1:
                self.assertEqual(len(tags), 7)
                checked_tags = [
                    tag for tag in tags if re.search(r"\bchecked(?:\s|=|>)", tag)
                ]
                self.assertEqual(len(checked_tags), 7, checked_tags)
            else:
                self.assertEqual(len(tags), 7)
                for tag in tags:
                    self.assertNotRegex(tag, r"\bchecked(?:\s|=|>)", tag)

        builder = map_builder_antigravity()
        builder.main_antigravity_validate_only_for_tests()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        embedded = re.search(
            r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>',
            rendered,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(embedded)
        assert embedded is not None
        embedded_graph = json.loads(html.unescape(embedded.group(1)))
        self.assertEqual(embedded_graph, graph)
        self.assertEqual(lock["represented_paths"], sorted(TRIPLET_ANTIGRAVITY))
        self.assertEqual(lock["source_parent_commit"], graph["source_parent_commit"])
        self.assertFalse(lock["working_tree_dirty"])
        self.assertEqual(graph["unknown_relationships"], [])
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
            "task2_predecessor",
            "task2_sources",
            "task2_config_tests",
            "task2_contract_tests",
            "task2_predecessor_tests",
            "task2_source_tests",
            "task3_gradual_signature_unknown",
            "task4_qualification_unknown",
            "task5_cli_unknown",
        ):
            self.assertIn(node_id, node_ids)
        for node in graph["nodes"]:
            if node["id"].endswith("_unknown"):
                self.assertIn("unknown", node["role"])
                self.assertIn("not implemented", node["role"])


if __name__ == "__main__":
    unittest.main()
