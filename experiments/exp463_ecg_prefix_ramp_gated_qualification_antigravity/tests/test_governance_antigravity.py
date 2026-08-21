from __future__ import annotations

import hashlib
import html
import html.parser
import importlib.util
import json
import re
import copy
import subprocess
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html"
)
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity-design.html"
)
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP463_PREREGISTRATION_antigravity.html",
    "docs/EXP463_RUNBOOK_antigravity.html",
    "docs/EXP463_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP463_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp463_antigravity.html",
    "docs/codemap/codemap_exp463_antigravity.json",
    "docs/codemap/codemap_exp463_antigravity.lock",
)
RUN_ROOT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/exp463_ecg_prefix_ramp_gated_qualification_antigravity_20260821"
)
RUN_IDENTITY_ANTIGRAVITY = (
    "exp463-ecg-prefix-ramp-gated-qualification-antigravity-20260821"
)
PACKAGE_ANTIGRAVITY = "experiments/exp463_ecg_prefix_ramp_gated_qualification_antigravity"
DESIGN_SHA256_ANTIGRAVITY = (
    "d5af337a09b11a707198eb0d669a9a6a4b330db44db698daa5ca218cdfbb7efb"
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


def map_builder_antigravity() -> object:
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp463_antigravity.py"
    if not path.is_file():
        raise AssertionError(f"missing map builder: {path}")
    spec = importlib.util.spec_from_file_location(
        "exp463_map_builder_antigravity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp463 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HtmlParserAntigravity(html.parser.HTMLParser):
    pass


class GovernanceAntigravityTest(unittest.TestCase):
    def _records_antigravity(self) -> tuple[dict[str, object], ...]:
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
            item for item in catalog["experiments"] if item.get("id") == "Exp463"
        ]
        champion_matches = [
            item for item in champion["history"] if item.get("experiment") == "Exp463"
        ]
        self.assertEqual(len(catalog_matches), 1)
        self.assertEqual(len(champion_matches), 1)
        return state["exp463"], catalog_matches[0], champion_matches[0]

    def test_task1_governance_antigravity(self) -> None:
        """Break caught: planned truth drifts before score-source implementation."""
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

        records = self._records_antigravity()
        expected_documents = {
            "preregistration": DOCUMENTS_ANTIGRAVITY[0],
            "runbook": DOCUMENTS_ANTIGRAVITY[1],
            "implementation_report": DOCUMENTS_ANTIGRAVITY[2],
            "dashboard": DOCUMENTS_ANTIGRAVITY[3],
        }
        expected_predecessor = {
            "experiment": "Exp462",
            "terminal_closeout_commit": "040422f62406e3ca19038b9609903c4ba76e1694",
            "status": "stopped_before_official_evaluation_qualification_failed",
            "primary_candidate": "hydra_period_gradual_rank_max",
            "primary_passed_count": 3,
            "primary_required_count": 10,
            "period_min_auc": 0.8960376741612595,
            "historical_period_folds": [
                {"seed": 20260721, "fold_index": 3, "period_change_auc": 0.8960376741612595},
                {"seed": 20261721, "fold_index": 4, "period_change_auc": 0.9285624465570143},
            ],
        }
        expected_contract = {
            "target_dataset": "ECG5000_normal_2",
            "data_split": "TRAIN_only",
            "train_shape": [1413, 140],
            "label_query_count_must_remain": 0,
            "official_command_invocations_must_remain": 0,
            "method_order": ["M0", "M1", "M2", "M3", "M4", "M5", "M6"],
            "method_definitions": {
                "M0": "Exp461 reviewed hydra_period_rank_max score",
                "M1": "front-prefix normalized one-dimensional ramp score",
                "M2": "M1 with 0.995-gated max against M0; sole primary candidate hydra_period_ramp_gated_max",
                "M3": "M1 with always-max combination against M0",
                "M4": "TRAIN-fit per-time median template residual ramp with 0.995-gated max",
                "M5": "whole-series median/MAD normalized ramp with 0.995-gated max",
                "M6": "Exp462 24D LocalGap replay control with always-max combination",
            },
            "primary_candidate": "hydra_period_ramp_gated_max",
            "seed_pair": [20260721, 20261721],
            "blocked_fold_count": 5,
            "onset": 56,
            "alpha": 0.005,
            "gate": 0.995,
            "per_type_auc_min": 0.85,
            "pooled_auc_min": 0.90,
            "normal_wilson_upper_max": 0.05,
            "gradual_drift_auc_min": 0.85,
            "gradual_drift_auc_strictly_above_exp462": 0.8402306222021025,
            "period_change_auc_global_min": 0.907392736783864,
            "historical_period_fold_thresholds": [
                {"seed": 20260721, "fold_index": 3, "period_change_auc_min": 0.907392736783864},
                {"seed": 20261721, "fold_index": 4, "period_change_auc_min": 0.9216148584075248},
            ],
            "test_metrics_may_inform_routing": False,
        }

        for record in records:
            with self.subTest(record=record["experiment"]):
                self.assertEqual(record["experiment"], "Exp463")
                self.assertEqual(record["kind"], "ecg_prefix_ramp_gated_qualification")
                self.assertEqual(record["status"], "planned_implementation_not_started")
                self.assertEqual(
                    record["implementation_status"],
                    "task_1_complete_pending_task_2",
                )
                self.assertEqual(record["run_identity"], RUN_IDENTITY_ANTIGRAVITY)
                self.assertEqual(record["run_root"], RUN_ROOT_ANTIGRAVITY)
                self.assertEqual(record["package"], PACKAGE_ANTIGRAVITY)
                self.assertEqual(
                    record["approved_design"], DESIGN_ANTIGRAVITY
                )
                self.assertEqual(record["approved_plan"], PLAN_ANTIGRAVITY)
                self.assertEqual(record["approved_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
                self.assertEqual(record["documents"], expected_documents)
                self.assertEqual(record["predecessor"], expected_predecessor)
                self.assertEqual(record["scientific_contract"], expected_contract)
                self.assertFalse(record["official_evaluate_started"])
                self.assertFalse(record["official_inspect_started"])
                self.assertEqual(record["official_evaluate_invocations"], 0)
                self.assertEqual(record["official_inspect_invocations"], 0)
                self.assertFalse(record["result_root_created"])
                for key in (
                    "official_result",
                    "prediction_sha256",
                    "evaluation_sha256",
                    "decision_sha256",
                    "artifact_map_sha256",
                    "dashboard_snapshot_sha256",
                ):
                    self.assertIsNone(record[key], key)
                self.assertFalse(record["test_labels_opened"])
                self.assertEqual(record["label_query_count"], 0)
                self.assertFalse(record["routing_allowed"])
                self.assertFalse(record["promotion_allowed"])
                self.assertFalse(record["pending_candidate_changed"])
                self.assertFalse(record["scientific_winner_selected"])
                self.assertFalse(record["review_only"])
                self.assertFalse(record["official_root_present"])

        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict) and isinstance(champion, dict)
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
        """Break caught: the governance surface or bounded future map drifts."""
        preregistration = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[0]).read_text(
            encoding="utf-8"
        )
        for phrase in (
            "planned_implementation_not_started",
            "Task 1/6",
            "hydra_period_ramp_gated_max",
            "M0",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "official command count 0",
            "label query count 0",
            "null",
            "0.995",
            "onset 56",
        ):
            self.assertIn(phrase, preregistration)

        runbook = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[1]).read_text(
            encoding="utf-8"
        )
        for phrase in (
            "Task 1 operator boundary",
            RUN_ROOT_ANTIGRAVITY,
            "must stay absent",
            "Do not run official evaluate",
            "Do not run official inspect",
            "official command count 0",
            "label query count 0",
        ):
            self.assertIn(phrase, runbook)

        report = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[2]).read_text(
            encoding="utf-8"
        )
        for phrase in (
            "Exp463 Task 1 governance report",
            "DONE_WITH_CONCERNS",
            "task_1_complete_pending_task_2",
            "RED",
            "GREEN",
            "official evaluate 0 / inspect 0",
            "label query count 0",
            DESIGN_SHA256_ANTIGRAVITY,
            "040422f62406e3ca19038b9609903c4ba76e1694",
            "Task 2–5",
        ):
            self.assertIn(phrase, report)

        dashboard = (ROOT_ANTIGRAVITY / DOCUMENTS_ANTIGRAVITY[3]).read_text(
            encoding="utf-8"
        )
        for phrase in (
            "Task 1",
            "task_1_complete_pending_task_2",
            "hydra_period_ramp_gated_max",
            "0.995",
            "56",
            "공식 실행 0회",
            "label query count",
            "artifact map SHA-256",
            "null",
            "Task 2",
            "Task 3",
            "Task 4",
            "Task 5",
        ):
            self.assertIn(phrase, dashboard)

        plan = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        for task_number in range(1, 7):
            task = re.search(
                rf"<section>\s*<h2><label><input type=\"checkbox\"[^>]*>Task {task_number}.*?</section>",
                plan,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(task, f"Task {task_number} section missing")
            assert task is not None
            tags = re.findall(r"<input\b[^>]*>", task.group(0), flags=re.IGNORECASE)
            self.assertEqual(len(tags), 7)
            for tag in tags:
                if task_number == 1:
                    self.assertRegex(tag, r"\bchecked(?:\s|=|>)", tag)
                else:
                    self.assertNotRegex(tag, r"\bchecked(?:\s|=|>)", tag)

        builder = map_builder_antigravity()
        builder.main_antigravity_validate_only_for_tests()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(
            encoding="utf-8"
        )
        self.assertIsInstance(graph, dict)
        self.assertIsInstance(lock, dict)
        assert isinstance(graph, dict) and isinstance(lock, dict)
        embedded = re.search(
            r'<pre id="codemap-json-antigravity" type="application/json">(.*?)</pre>',
            rendered,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(embedded)
        assert embedded is not None
        self.assertEqual(json.loads(html.unescape(embedded.group(1))), graph)
        self.assertEqual(graph["schema"], "exp463_task1_governance_codemap_v1")
        self.assertEqual(lock["schema"], "exp463_task1_governance_codemap_lock_v1")
        self.assertEqual(lock["represented_paths"], sorted(TRIPLET_ANTIGRAVITY))
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_ANTIGRAVITY, text=True
        ).strip()
        parents = subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
            cwd=ROOT_ANTIGRAVITY,
            text=True,
        ).strip().split()[1:]
        self.assertIn(lock["source_parent_commit"], {head, *parents})
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
            "task2_unknown",
            "task3_unknown",
            "task4_unknown",
            "task5_unknown",
        ):
            self.assertIn(node_id, node_ids)
        for node in graph["nodes"]:
            if node["id"].endswith("_unknown"):
                self.assertIn("unknown", node["role"])
                self.assertIn("not implemented", node["role"])
        for edge in graph["edges"]:
            self.assertIn(edge["from"], node_ids)
            self.assertIn(edge["to"], node_ids)
            self.assertIn(edge["type"], {"governs", "tests"})
            self.assertTrue(edge["evidence"]["path"])
            self.assertTrue(edge["evidence"]["symbol"])
        for node in graph["nodes"]:
            for evidence in node["evidence"]:
                evidence_path = ROOT_ANTIGRAVITY / evidence["path"]
                self.assertTrue(evidence_path.is_file(), evidence["path"])
                self.assertIn(
                    evidence["symbol"],
                    evidence_path.read_text(encoding="utf-8"),
                    f"unresolvable node evidence: {evidence}",
                )
        for edge in graph["edges"]:
            evidence = edge["evidence"]
            evidence_path = ROOT_ANTIGRAVITY / evidence["path"]
            self.assertIn(
                evidence["symbol"],
                evidence_path.read_text(encoding="utf-8"),
                f"unresolvable edge evidence: {evidence}",
            )

    def test_map_fingerprint_drift_is_rejected_antigravity(self) -> None:
        """Break caught: a stale lock must not be accepted by path-only checks."""
        builder = map_builder_antigravity()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(
            encoding="utf-8"
        )
        self.assertIsInstance(graph, dict)
        self.assertIsInstance(lock, dict)
        assert isinstance(graph, dict) and isinstance(lock, dict)
        builder.validate_map_antigravity(graph, lock, rendered)
        stale_tracked = copy.deepcopy(lock)
        stale_tracked["tracked_fingerprints"]["RESEARCH_STATE.json"] = "0" * 64
        with self.assertRaises(AssertionError):
            builder.validate_map_antigravity(graph, stale_tracked, rendered)
        stale_module = copy.deepcopy(lock)
        stale_module["module_fingerprints"]["docs"] = "0" * 64
        with self.assertRaises(AssertionError):
            builder.validate_map_antigravity(graph, stale_module, rendered)

    def test_review_scratch_is_stable_but_tracked_source_drifts_antigravity(self) -> None:
        """Break caught: ignored review scratch is non-scientific, tracked source is not."""
        builder = map_builder_antigravity()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(
            encoding="utf-8"
        )
        self.assertIsInstance(graph, dict)
        self.assertIsInstance(lock, dict)
        assert isinstance(graph, dict) and isinstance(lock, dict)
        builder.validate_map_antigravity(graph, lock, rendered)

        review_artifact = (
            ROOT_ANTIGRAVITY
            / ".superpowers/sdd/2026-08-21-exp463-ecg-prefix-ramp-gated-qualification-antigravity.html/"
            "fingerprint-regression-antigravity.html"
        )
        tracked_source = ROOT_ANTIGRAVITY / "RESEARCH_STATE.json"
        original_source = tracked_source.read_bytes()
        try:
            review_artifact.write_text("review scratch v1\n", encoding="utf-8")
            builder.validate_map_antigravity(graph, lock, rendered)
            review_artifact.write_text("review scratch v2\n", encoding="utf-8")
            builder.validate_map_antigravity(graph, lock, rendered)

            tracked_source.write_bytes(original_source + b"\n")
            with self.assertRaises(AssertionError):
                builder.validate_map_antigravity(graph, lock, rendered)
        finally:
            tracked_source.write_bytes(original_source)
            if review_artifact.exists():
                review_artifact.unlink()


if __name__ == "__main__":
    unittest.main()
