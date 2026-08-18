"""Governance regression for Exp459 Task 1 planned-not-started truth."""

from __future__ import annotations

import hashlib
import html.parser
import importlib.util
import json
import copy
import re
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PLAN_ANTIGRAVITY = (
    "docs/superpowers/plans/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity.html"
)
DESIGN_ANTIGRAVITY = (
    "docs/superpowers/specs/2026-08-18-exp459-ecg-normal2-blocked-validation-antigravity-design.html"
)
DOCUMENTS_ANTIGRAVITY = (
    "docs/EXP459_PREREGISTRATION_antigravity.html",
    "docs/EXP459_RUNBOOK_antigravity.html",
    "docs/EXP459_IMPLEMENTATION_REPORT_antigravity.html",
    "docs/EXP459_DASHBOARD_antigravity.html",
)
TRIPLET_ANTIGRAVITY = (
    "docs/codemap/codemap_exp459_antigravity.html",
    "docs/codemap/codemap_exp459_antigravity.json",
    "docs/codemap/codemap_exp459_antigravity.lock",
)
DESIGN_SHA256_ANTIGRAVITY = (
    "47d530a0cbe7f7cdb0b85f2c0c546b7ede158d8a85824eba8c8caece16b2e275"
)
PLAN_SHA256_ANTIGRAVITY = (
    "d0a2a2643593b973b4ec8ac1f24a43fdd109916b1f73956dea9bc4cffcb572ef"
)
RUN_IDENTITY_ANTIGRAVITY = (
    "exp459-ecg-normal2-blocked-validation-antigravity-20260818"
)
IMPLEMENTATION_STATUS_ANTIGRAVITY = (
    "task_4_evaluation_artifacts_replay_dashboard_complete"
)
RUN_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp459_ecg_normal2_blocked_validation_antigravity_20260818"
)
PACKAGE_ANTIGRAVITY = "experiments/exp459_ecg_normal2_blocked_validation_antigravity"
PENDING_CANDIDATE_ANTIGRAVITY = {
    "evaluation": None,
    "experiment": None,
    "prediction_sha256": None,
    "status": "none_exp446_publication_inspected_complete_not_pending_champion",
}
CHAMPION_SNAPSHOT_ANTIGRAVITY = {
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
    path = ROOT_ANTIGRAVITY / "scratch/build_codemap_exp459_antigravity.py"
    self = path.is_file()
    if not self:
        raise AssertionError(f"missing map builder: {path}")
    spec = importlib.util.spec_from_file_location(
        "exp459_map_builder_antigravity",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp459 map builder is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HtmlTextExtractorAntigravity(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def unresolved_evidence_antigravity(graph: dict[str, object]) -> list[dict[str, str]]:
    unresolved: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for collection in ("nodes", "edges"):
        for item in graph[collection]:
            for evidence in item["evidence"] if collection == "nodes" else [item["evidence"]]:
                key = (evidence["path"], evidence["symbol"])
                if key in seen:
                    continue
                seen.add(key)
                path = ROOT_ANTIGRAVITY / evidence["path"]
                if not path.is_file():
                    unresolved.append({"path": evidence["path"], "symbol": evidence["symbol"]})
                    continue
                text = path.read_text(encoding="utf-8")
                if path.suffix == ".html":
                    parser = HtmlTextExtractorAntigravity()
                    parser.feed(text)
                    haystack = " ".join(parser.parts)
                else:
                    haystack = text
                if evidence["symbol"] not in haystack:
                    unresolved.append({"path": evidence["path"], "symbol": evidence["symbol"]})
    return unresolved


class GovernanceAntigravityTest(unittest.TestCase):
    def test_task1_documents_triplet_and_checkboxes_antigravity(self) -> None:
        """Break caught: Task 2 claims progress without HTML/docs/map truth."""
        for relative in DOCUMENTS_ANTIGRAVITY + TRIPLET_ANTIGRAVITY:
            with self.subTest(relative=relative):
                path = ROOT_ANTIGRAVITY / relative
                self.assertTrue(path.is_file(), relative)
                self.assertIn("antigravity", path.name)
        for relative in DOCUMENTS_ANTIGRAVITY:
            path = ROOT_ANTIGRAVITY / relative
            self.assertEqual(path.suffix, ".html")
            self.assertFalse(path.with_suffix(".md").exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", text.lower())
            self.assertIn("Exp459", text)

        plan = (ROOT_ANTIGRAVITY / PLAN_ANTIGRAVITY).read_text(encoding="utf-8")
        for task_number in range(1, 7):
            section = re.search(
                rf"<h2>Task {task_number}:.*?(?=<h2>Task {task_number + 1}:|<h2>전체 검증 명령 모음)",
                plan,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(section)
            assert section is not None
            tags = re.findall(r"<input\b[^>]*>", section.group(0), flags=re.IGNORECASE)
            checked = [tag for tag in tags if re.search(r"\bchecked(?:\s|=|>)", tag)]
            if task_number in {1, 2, 3, 4}:
                self.assertEqual(len(checked), len(tags))
            else:
                self.assertEqual(len(checked), 0)
        status_rows = re.findall(r"<li><label><input\b[^>]*>(Task \d 완료)</label></li>", plan)
        self.assertEqual(status_rows, [f"Task {n} 완료" for n in range(1, 7)])
        initial_checked = re.findall(
            r"<li><label><input\b[^>]*checked[^>]*>(Task \d 완료)</label></li>",
            plan,
            flags=re.IGNORECASE,
        )
        self.assertEqual(initial_checked, ["Task 1 완료", "Task 2 완료", "Task 3 완료", "Task 4 완료"])

    def test_exp459_planned_truth_and_non_promotion_antigravity(self) -> None:
        """Break caught: Exp459 records an official result, changes champion, or loses frozen hashes."""
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        catalog = strict_json_antigravity(ROOT_ANTIGRAVITY / "EXPERIMENT_CATALOG.json")
        champion = strict_json_antigravity(ROOT_ANTIGRAVITY / "CHAMPION_LEDGER.json")
        assert isinstance(state, dict)
        assert isinstance(catalog, dict)
        assert isinstance(champion, dict)

        exp459_state = state["exp459"]
        catalog_entry = next(item for item in catalog["experiments"] if item["id"] == "Exp459")
        champion_entry = next(item for item in champion["history"] if item["experiment"] == "Exp459")
        self.assertEqual(sum(item["id"] == "Exp459" for item in catalog["experiments"]), 1)
        self.assertEqual(sum(item["experiment"] == "Exp459" for item in champion["history"]), 1)

        for record in (exp459_state, catalog_entry, champion_entry):
            with self.subTest(record=record["experiment"]):
                self.assertEqual(record["experiment"], "Exp459")
                self.assertEqual(record["kind"], "ecg_normal2_blocked_validation")
                self.assertEqual(record["status"], "planned_not_started")
                self.assertEqual(record["implementation_status"], IMPLEMENTATION_STATUS_ANTIGRAVITY)
                self.assertEqual(record["run_identity"], RUN_IDENTITY_ANTIGRAVITY)
                self.assertEqual(record["run_root"], str(RUN_ROOT_ANTIGRAVITY))
                self.assertEqual(record["package"], PACKAGE_ANTIGRAVITY)
                self.assertEqual(record["approved_design"], DESIGN_ANTIGRAVITY)
                self.assertEqual(record["approved_design_sha256"], DESIGN_SHA256_ANTIGRAVITY)
                self.assertEqual(record["approved_plan"], PLAN_ANTIGRAVITY)
                self.assertEqual(record["approved_plan_sha256"], PLAN_SHA256_ANTIGRAVITY)
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
                self.assertEqual(record["label_query_count"], 0)
                self.assertFalse(record["test_labels_opened"])
                self.assertFalse(record["routing_allowed"])
                self.assertFalse(record["promotion_allowed"])
                self.assertFalse(record["pending_candidate_changed"])
                self.assertFalse(record["scientific_winner_selected"])
                self.assertEqual(
                    record["predecessor"],
                    {
                        "experiment": "Exp458",
                        "reviewed_commit": "fc530cd2b392394aa7ffbd1408bc06d5b1b7c806",
                        "artifact_map_sha256": (
                            "7ef36f232db5d2f2396f3dcdeda15af2e93c34e18a6e05dc411e37fe855737f9"
                        ),
                    },
                )
                self.assertEqual(
                    record["scientific_contract"],
                    {
                        "target_dataset": "ECG5000_normal_2",
                        "hydra_seeds": [20260721, 20261721],
                        "fold_sizes": [283, 283, 283, 282, 282],
                        "fold_count": 5,
                        "primary_types": [
                            "level_shift",
                            "gradual_drift",
                            "amplitude_scaling",
                            "period_change",
                        ],
                        "per_type_auc_min": 0.85,
                        "pooled_auc_min": 0.90,
                        "normal_wilson_upper_max": 0.05,
                        "shuffle_allowed": False,
                        "final_decision_rule": "all_10_fold_seed_records_must_pass_for_stable",
                    },
                )

        self.assertEqual(state["pending_candidate"], PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(champion["pending_candidate"], PENDING_CANDIDATE_ANTIGRAVITY)
        self.assertEqual(
            {
                "champion_status": champion["champion_status"],
                "pending_candidate": champion["pending_candidate"],
                "verified_references": champion["verified_references"],
            },
            CHAMPION_SNAPSHOT_ANTIGRAVITY,
        )
        self.assertEqual(
            hashlib.sha256((ROOT_ANTIGRAVITY / DESIGN_ANTIGRAVITY).read_bytes()).hexdigest(),
            DESIGN_SHA256_ANTIGRAVITY,
        )
        self.assertFalse(RUN_ROOT_ANTIGRAVITY.exists())

    def test_exp459_map_is_bounded_and_validates_antigravity(self) -> None:
        """Break caught: Task 1 map drifts from HTML/JSON/lock parity or loses governance coverage."""
        for relative in TRIPLET_ANTIGRAVITY:
            self.assertTrue((ROOT_ANTIGRAVITY / relative).is_file(), relative)
        builder = map_builder_antigravity()
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        lock = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[2])
        rendered = (ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[0]).read_text(encoding="utf-8")
        assert isinstance(graph, dict)
        self.assertLessEqual(len(graph["nodes"]), 32)
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertIn("governance_regression", node_by_id)
        self.assertIn("task2_config", node_by_id)
        self.assertIn("task2_contracts", node_by_id)
        self.assertIn("task2_blocked_folds", node_by_id)
        self.assertIn("task3_sources", node_by_id)
        self.assertIn("task3_qualification", node_by_id)
        self.assertIn("task4_evaluation", node_by_id)
        self.assertIn("task4_artifacts", node_by_id)
        self.assertIn("task4_replay", node_by_id)
        self.assertIn("task4_dashboard_export", node_by_id)
        self.assertIn("research_state", node_by_id)
        self.assertIn("experiment_catalog", node_by_id)
        self.assertIn("champion_ledger", node_by_id)
        self.assertTrue(
            any(
                edge["from"] == "governance_regression"
                and edge["to"] == "research_state"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "governance_regression"
                and edge["to"] == "champion_ledger"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task2_contract_tests"
                and edge["to"] == "task2_contracts"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task3_source_tests"
                and edge["to"] == "task3_sources"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task3_qualification_tests"
                and edge["to"] == "task3_qualification"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_evaluation_tests"
                and edge["to"] == "task4_evaluation"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_artifact_tests"
                and edge["to"] == "task4_artifacts"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_replay_tests"
                and edge["to"] == "task4_replay"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertTrue(
            any(
                edge["from"] == "task4_dashboard_tests"
                and edge["to"] == "task4_dashboard_export"
                and edge["type"] == "tests"
                for edge in graph["edges"]
            )
        )
        self.assertEqual(graph["unknown_relationships"], [])
        builder.validate_map_antigravity(graph, lock, rendered)

    def test_exp459_map_evidence_symbols_resolve_in_cited_sources_antigravity(self) -> None:
        """Break caught: map evidence cites HTML or Python symbols that do not resolve in the cited file."""
        graph = strict_json_antigravity(ROOT_ANTIGRAVITY / TRIPLET_ANTIGRAVITY[1])
        assert isinstance(graph, dict)
        self.assertEqual(unresolved_evidence_antigravity(graph), [])

    def test_exp459_validator_rejects_unresolved_evidence_symbol_antigravity(self) -> None:
        """Break caught: validator accepts an evidence symbol that does not resolve in its cited source."""
        builder = map_builder_antigravity()
        source_commit = builder.subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_ANTIGRAVITY,
            text=True,
        ).strip()
        graph, lock, rendered = builder.build_map_antigravity(
            source_commit,
            "2026-08-18T00:00:00Z",
        )
        graph = copy.deepcopy(graph)
        graph["edges"][0]["evidence"]["symbol"] = "missing_symbol_antigravity"
        rendered = builder.html_antigravity(graph)
        with self.assertRaisesRegex(ValueError, "unresolved evidence"):
            builder.validate_map_antigravity(graph, lock, rendered)


if __name__ == "__main__":
    unittest.main()
