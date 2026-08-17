from __future__ import annotations

import html.parser
import importlib.util
import json
from pathlib import Path
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP457_PREREGISTRATION_antigravity_KO.html",
    "docs/EXP457_RUNBOOK_antigravity_KO.html",
    "docs/EXP457_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP457_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp457_antigravity.html",
    "docs/codemap/codemap_exp457_antigravity.json",
    "docs/codemap/codemap_exp457_antigravity.lock",
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
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def test_exp457_scientific_contract_antigravity(self) -> None:
        """Break caught: planned Exp457 truth drifts from the approved result gates."""
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
            state["exp457"],
            next(item for item in catalog["experiments"] if item["id"] == "Exp457"),
            next(item for item in champion["history"] if item["experiment"] == "Exp457"),
        )
        for record in records:
            self.assertEqual(record["kind"], "ecg_train_synthetic_confidence_routing")
            self.assertEqual(
                record["status"],
                "stopped_before_official_evaluation_train_gate_failed",
            )
            self.assertEqual(
                record["implementation_status"],
                "task_5_candidate_verification_complete",
            )
            self.assertEqual(record["approved_design_sha256"], "3b702e0224dd86e1d0c93268b8a2e2691ea3858cbd30a4792f9c6539c0424e20")
            config = record["scientific_contract"]
            self.assertEqual(config["target_family"], "ECG5000")
            self.assertEqual(config["candidate"], "hydra_local_gap")
            self.assertEqual(config["chlorine_route"], "multirocket_local_gap_for_all_three_datasets")
            self.assertEqual(config["hydra_seeds"], [20260721, 20261721])
            self.assertEqual(config["folds"], 5)
            self.assertEqual(config["alpha"], 0.005)
            self.assertEqual(config["knn_k"], 3)
            self.assertEqual(config["kernels"], 8)
            self.assertEqual(config["groups"], 64)
            self.assertEqual(config["synthetic_primary_types"], ["level_shift", "gradual_drift", "amplitude_scaling", "period_change"])
            self.assertEqual(config["pooled_sensitivity_min"], 0.70)
            self.assertEqual(config["per_type_sensitivity_min"], 0.50)
            self.assertEqual(config["normal_wilson_upper_max"], 0.05)
            self.assertIs(config["finite_required"], True)
            self.assertEqual(config["structure_gate"], "added_count > removed_count")
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
            for key in ("prediction_sha256", "evaluation_sha256", "decision_sha256", "artifact_map_sha256", "dashboard_snapshot_sha256"):
                self.assertIsNone(record[key])
            self.assertFalse(record["routing_allowed"])
            self.assertFalse(record["promotion_allowed"])
            self.assertFalse(record["pending_candidate_changed"])
            self.assertIs(record["test_labels_opened"], False)
            self.assertIs(type(record["label_query_count"]), int)
            self.assertEqual(record["label_query_count"], 0)
            self.assertIs(record["scientific_winner_selected"], False)
        self.assertEqual(state["pending_candidate"], FROZEN_PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], FROZEN_PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(
            {key: champion[key] for key in FROZEN_CHAMPION_ANTIGRAVITY},
            FROZEN_CHAMPION_ANTIGRAVITY,
        )

    def test_dashboard_plan_and_map_expose_task5_candidate_without_official_results_antigravity(self) -> None:
        """Break caught: Task 5 fabricates an official result or leaves the CLI boundary future."""
        dashboard = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(encoding="utf-8")
        for value in (
            "Task 5/6",
            "stopped_before_official_evaluation_train_gate_failed",
            "아직 공식 결과 없음",
            "always_baseline</td><td>837</td><td>66</td><td>0.5742574257425742",
            "family_wide_local_gap</td><td>835</td><td>63</td><td>0.5764150943396227",
            "oracle_per_dataset</td><td>828</td><td>64</td><td>0.5808270676691729",
            "ecg_train_confidence_router</td><td>836</td><td>64</td><td>0.5754716981132075",
            "real TRAIN candidate qualification reproduced exactly twice outside the official result root",
            "Gate 1 numeric evidence and evaluation provenance are semantically recomputed",
            "사용자가 공식 TEST 평가를 실행하지 않기로 결정",
            "FN 836 / FP 64 / F1 0.5754716981132075",
            "공식 aeon Hydra는 CPU 구현을 유지하며 MLX는 적용하지 않는다.",
        ):
            self.assertIn(value, dashboard)

        report = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]).read_text(encoding="utf-8")
        self.assertIn(
            "Gate 1 numeric evidence and evaluation provenance are semantically recomputed",
            report,
        )

        plan = (ROOT_ANTIGRAVITY / "docs/superpowers/plans/2026-08-18-exp457-ecg-train-synthetic-confidence-routing-antigravity.html").read_text(encoding="utf-8")
        self.assertIn('<input type="checkbox" checked disabled>Task 5:', plan)
        task5 = plan.split("Task 5:", 1)[1].split("Task 6:", 1)[0]
        self.assertEqual(task5.count('<input type="checkbox" checked disabled>'), 7)
        task6 = plan.split("Task 6:", 1)[1].split('</section>', 1)[0]
        self.assertEqual(task6.count('<input type="checkbox" disabled>'), 8)
        self.assertEqual(task6.count('<input type="checkbox" checked disabled>'), 1)
        self.assertIn(
            "종료 결정: TRAIN Gate 실패로 사용자가 공식 TEST 평가를 실행하지 않음",
            task6,
        )

        spec = importlib.util.spec_from_file_location(
            "exp457_map_builder_antigravity",
            ROOT_ANTIGRAVITY / "scratch/build_codemap_exp457_antigravity.py",
        )
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict)
        nodes = {node["id"]: node for node in graph["nodes"]}
        for node_id in (
            "dashboard",
            "research_state",
            "governance_regression",
            "map_builder",
            "frozen_config",
            "result_contracts",
            "primary_perturbations",
            "train_source",
            "train_qualification",
            "router",
            "evaluation",
            "artifacts",
            "replay",
            "dashboard_export",
            "source_regression",
            "qualification_regression",
            "task4_regressions",
            "cli",
            "cli_launcher",
            "cli_regression",
        ):
            self.assertNotEqual(nodes[node_id].get("status"), "future_unknown")
        for node_id in (
            "frozen_config",
            "result_contracts",
            "primary_perturbations",
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
            self.assertTrue(nodes[node_id]["evidence"])
            self.assertTrue(nodes[node_id]["tests"])
            self.assertTrue((ROOT_ANTIGRAVITY / nodes[node_id]["path"]).is_file())
        self.assertNotIn("future_cli", nodes)
        builder.validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    unittest.main()
