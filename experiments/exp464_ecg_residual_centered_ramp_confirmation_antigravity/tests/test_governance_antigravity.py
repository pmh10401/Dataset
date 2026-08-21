from __future__ import annotations

import hashlib
import html.parser
import importlib.util
import json
import re
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
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP464_DASHBOARD_antigravity.html",
    "docs/EXP464_PREREGISTRATION_antigravity.html",
    "docs/EXP464_RUNBOOK_antigravity.html",
    "docs/EXP464_IMPLEMENTATION_REPORT_antigravity.html",
)
FOUNDATION_FILES_ANTIGRAVITY = (
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/config_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/contracts_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/predecessor_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/sources_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_config_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_contracts_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_predecessor_antigravity.py",
    "experiments/exp464_ecg_residual_centered_ramp_confirmation_antigravity/tests/test_sources_antigravity.py",
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
PRIMARY_CANDIDATE_ANTIGRAVITY = (
    "template_residual_centered_ramp_confirmed_max"
)
STATUS_ANTIGRAVITY = "task_2_complete_pending_task_3"
METHOD_ORDER_ANTIGRAVITY = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
EXPECTED_CHECKBOX_COUNT_ANTIGRAVITY = 42
EXPECTED_PENDING_ANTIGRAVITY = {
    "evaluation": None,
    "experiment": None,
    "prediction_sha256": None,
    "status": "none_exp446_publication_inspected_complete_not_pending_champion",
}


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

    def _records_antigravity(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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

    def test_exp464_is_planned_without_results_antigravity(self) -> None:
        """Break caught: Exp464 governance truth drifts before science code exists."""
        dashboard = ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[0]
        self.assertTrue(dashboard.exists())

        missing = [
            relative
            for relative in DOCUMENTS_ANTIGRAVITY[1:] + FOUNDATION_FILES_ANTIGRAVITY + TRIPLET_ANTIGRAVITY
            if not (ROOT_ANTIGRAVITY / relative).is_file()
        ]
        self.assertEqual(missing, [], f"missing Task 1 artifacts: {missing}")
        for relative in DOCUMENTS_ANTIGRAVITY + FOUNDATION_FILES_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertTrue(path.is_file(), relative)
            self.assertIn("antigravity", path.name)

        self.assertEqual(
            hashlib.sha256((ROOT_ANTIGRAVITY / DESIGN_ANTIGRAVITY).read_bytes()).hexdigest(),
            DESIGN_SHA256_ANTIGRAVITY,
        )
        run_root = Path(RUN_ROOT_ANTIGRAVITY)
        self.assertFalse(run_root.exists(), RUN_ROOT_ANTIGRAVITY)
        self.assertFalse(run_root.is_symlink(), RUN_ROOT_ANTIGRAVITY)

        state_record, catalog_record, champion_record = self._records_antigravity()
        expected_documents = {
            "dashboard": DOCUMENTS_ANTIGRAVITY[0],
            "preregistration": DOCUMENTS_ANTIGRAVITY[1],
            "runbook": DOCUMENTS_ANTIGRAVITY[2],
            "implementation_report": DOCUMENTS_ANTIGRAVITY[3],
        }
        expected_predecessor = {
            "experiment": "Exp463",
            "terminal_closeout_commit": "147225831e9c13e9df5043fa88757bb03e01b79a",
            "status": "stopped_before_official_evaluation_qualification_failed",
            "primary_candidate": "hydra_period_ramp_gated_max",
            "primary_passed_count": 7,
            "primary_required_count": 10,
            "gradual_min_auc": 0.83450279161008,
            "period_min_auc": 0.8984897640963734,
            "historical_period_folds": [
                {"seed": 20260721, "fold_index": 3, "period_change_auc": 0.907392736783864},
                {"seed": 20261721, "fold_index": 4, "period_change_auc": 0.9216148584075248},
            ],
        }
        expected_contract = {
            "target_dataset": "ECG5000_normal_2",
            "data_split": "TRAIN_only",
            "train_shape": [1413, 140],
            "label_query_count_must_remain": 0,
            "official_command_invocations_must_remain": 0,
            "method_order": list(METHOD_ORDER_ANTIGRAVITY),
            "primary_candidate": PRIMARY_CANDIDATE_ANTIGRAVITY,
            "seed_pair": [20260721, 20261721],
            "blocked_fold_count": 5,
            "onset": 56,
            "cosine_min": 0.95,
            "per_type_auc_min": 0.85,
            "pooled_auc_min": 0.90,
            "normal_wilson_upper_max": 0.05,
            "gradual_drift_auc_min": 0.85,
            "gradual_drift_auc_strictly_above_exp463": 0.8402306222021025,
            "period_change_auc_global_min": 0.907392736783864,
            "historical_period_fold_thresholds": [
                {"seed": 20260721, "fold_index": 3, "period_change_auc_min": 0.907392736783864},
                {"seed": 20261721, "fold_index": 4, "period_change_auc_min": 0.9216148584075248},
            ],
            "test_metrics_may_inform_routing": False,
        }
        expected_method_definitions = {
            "C0": "Exp463 M0 exact replay control",
            "C1": "Exp463 M2 exact replay control",
            "C2": "Exp463 M4 exact replay control",
            "C3": "template residual centered ramp rank only without M0 max or cosine gate",
            "C4": "same C3 rank with unconditional max against C0 and no cosine gate",
            "C5": "centered ramp confirmation without template subtraction",
            "C6": "template residual centered ramp rank with cosine >= 0.95 confirmed max against C0",
            "C7": "Exp463 M6 exact replay control",
            "C8": "template residual ramp confirmation without centering",
        }

        for record in (state_record, catalog_record, champion_record):
            self.assertEqual(record["experiment"], "Exp464")
            self.assertEqual(
                record["kind"],
                "ecg_residual_centered_ramp_confirmation_qualification",
            )
            self.assertEqual(record["status"], STATUS_ANTIGRAVITY)
            self.assertEqual(record["implementation_status"], STATUS_ANTIGRAVITY)
            self.assertEqual(
                record["run_identity"],
                "exp464-ecg-residual-centered-ramp-confirmation-antigravity-20260821",
            )
            self.assertEqual(record["run_root"], RUN_ROOT_ANTIGRAVITY)
            self.assertEqual(record["package"], PACKAGE_ANTIGRAVITY)
            self.assertEqual(record["approved_design"], DESIGN_ANTIGRAVITY)
            self.assertEqual(record["approved_plan"], PLAN_ANTIGRAVITY)
            self.assertEqual(record["approved_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
            self.assertEqual(record["documents"], expected_documents)
            self.assertEqual(record["predecessor"], expected_predecessor)
            self.assertEqual(record["scientific_contract"], expected_contract)
            self.assertEqual(record["method_definitions"], expected_method_definitions)
            self.assertEqual(record["official_evaluate_started"], False)
            self.assertEqual(record["official_inspect_started"], False)
            self.assertEqual(record["official_evaluate_count"], 0)
            self.assertEqual(record["official_inspect_count"], 0)
            self.assertEqual(record["official_evaluate_invocations"], 0)
            self.assertEqual(record["official_inspect_invocations"], 0)
            self.assertEqual(record["private_run_count"], 0)
            self.assertEqual(record["result_root_created"], False)
            self.assertIsNone(record["official_result"])
            self.assertIsNone(record["prediction_sha256"])
            self.assertIsNone(record["evaluation_sha256"])
            self.assertIsNone(record["decision_sha256"])
            self.assertIsNone(record["artifact_map_sha256"])
            self.assertIsNone(record["dashboard_snapshot_sha256"])
            self.assertEqual(record["test_labels_opened"], False)
            self.assertEqual(record["label_query_count"], 0)
            self.assertEqual(record["routing_allowed"], False)
            self.assertEqual(record["promotion_allowed"], False)
            self.assertEqual(record["pending_candidate_changed"], False)
            self.assertEqual(record["scientific_winner_selected"], False)
            self.assertEqual(record["official_root_present"], False)
            self.assertEqual(record["review_only"], False)

        champion_ledger = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        self.assertEqual(champion_ledger["pending_candidate"], EXPECTED_PENDING_ANTIGRAVITY)

        dashboard_text = dashboard.read_text(encoding="utf-8")
        self.assertIn(STATUS_ANTIGRAVITY, dashboard_text)
        self.assertIn("M0 9/10", dashboard_text)
        self.assertIn("M1 0/10", dashboard_text)
        self.assertIn("M2 7/10", dashboard_text)
        self.assertIn("M4 9/10", dashboard_text)
        self.assertIn("진단용", dashboard_text)
        self.assertIn("config_antigravity.py", dashboard_text)

        for relative in DOCUMENTS_ANTIGRAVITY:
            parser = HtmlParserAntigravity()
            parser.feed((ROOT_ANTIGRAVITY / relative).read_text(encoding="utf-8"))
            parser.close()

        plan_parser = CheckboxParserAntigravity()
        plan_parser.feed((ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8"))
        plan_parser.close()
        self.assertEqual(plan_parser.checkbox_count, EXPECTED_CHECKBOX_COUNT_ANTIGRAVITY)
        self.assertEqual(plan_parser.checked_count, 14)

        builder_module = load_builder_module_antigravity()
        build_result = builder_module.build_triplet_antigravity(validate_only=True)
        self.assertIsInstance(build_result, dict)
        self.assertEqual(build_result["triplet_paths"], list(TRIPLET_ANTIGRAVITY))
        self.assertEqual(build_result["source_parent"], build_result["allowed_source_commits"][0])

        subprocess.run(
            ["python3", "scratch/build_codemap_exp464_antigravity.py", "--validate-only"],
            cwd=ROOT_ANTIGRAVITY,
            check=True,
            capture_output=True,
            text=True,
        )
        for relative in TRIPLET_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertTrue(path.is_file(), relative)

        map_json = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        map_lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        self.assertEqual(map_lock["source_parent"], build_result["source_parent"])
        self.assertEqual(map_lock["generated_files"], list(TRIPLET_ANTIGRAVITY))
        self.assertEqual(map_lock["fingerprint_count"], len(map_lock["tracked_fingerprints"]))
        self.assertGreater(map_json["node_count"], 0)
        self.assertGreater(map_json["edge_count"], 0)
        self.assertEqual(map_json["graph_schema"], "exp464_task2_foundation_codemap_v1")
        self.assertEqual(map_lock["lock_schema"], "exp464_task2_foundation_codemap_lock_v1")
        node_ids = {node["id"] for node in map_json["nodes"]}
        self.assertIn("task2_config", node_ids)
        self.assertIn("task2_contracts", node_ids)
        self.assertIn("task2_predecessor", node_ids)
        self.assertIn("task2_sources", node_ids)
        self.assertIn("future_task3_scoring", node_ids)

        html_text = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        match = re.search(
            r'<script id="graph-data" type="application/json">(.*?)</script>',
            html_text,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        assert match is not None
        embedded_json = json.loads(match.group(1))
        self.assertEqual(embedded_json, map_json)


if __name__ == "__main__":
    unittest.main()
